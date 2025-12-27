# m.py (정리된 최종본)
# - 환승역 난이도(거리+시간) 점수화
# - 최단시간 경로 vs 환승쉬움(피로도 반영) 경로 추천 (Dijkstra)
# - 시각화 3종: TOP20, 분포, 거리-시간 산점도
#
# 필요 라이브러리: pandas numpy requests matplotlib
# pip install pandas numpy requests matplotlib

import heapq
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

# =========================
# 0) matplotlib 한글 폰트
# =========================
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 1) 설정
# =========================
API_KEY = "56e21c600b9e5ffd8366b116ad20ab56f8c78ec0a83357203e15890f9e33d6b4"  # ⚠️ 키 노출 방지 위해 가급적 재발급 권장
TRANSFER_URL = "https://api.odcloud.kr/api/15044419/v1/uddi:7008c675-928f-41d6-9a01-b3541f78466b"  # ✅ 쿼리 제거
SEGMENTS_CSV = "data/segments.csv"

# “편한 경로” 강도 조절 (alpha ↑ = 환승 어려움 더 회피)
ALPHA = 0.2

HEADERS = {"Authorization": f"Infuser {API_KEY}"}


# =========================
# 2) 환승 데이터 가져오기
# =========================
def fetch_all_transfer(per_page=100):
    page = 1
    all_rows = []
    total = None

    while True:
        params = {"page": page, "perPage": per_page, "returnType": "JSON"}
        r = requests.get(TRANSFER_URL, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()

        j = r.json()
        rows = j.get("data", [])
        if not rows:
            break

        all_rows.extend(rows)
        if total is None:
            total = j.get("totalCount")

        if total is not None and len(all_rows) >= int(total):
            break

        page += 1

    return pd.DataFrame(all_rows)


def mmss_to_sec(s):
    # "02:13" -> 133
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    if ":" not in s:
        return np.nan
    m, sec = s.split(":")
    return int(m) * 60 + int(sec)


def minmax(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    mn, mx = x.min(), x.max()
    if mx == mn:
        return x * 0
    return (x - mn) / (mx - mn)


def clean_transfer(df_raw: pd.DataFrame):
    df = df_raw.copy()

    # 환승거리 컬럼 통일: "환승거리" vs "환승 거리"
    dist = pd.to_numeric(df.get("환승거리"), errors="coerce")
    if "환승 거리" in df.columns:
        dist = dist.fillna(pd.to_numeric(df["환승 거리"], errors="coerce"))
    df["transfer_dist_m"] = dist

    # 환승시간(mm:ss) -> sec
    df["transfer_time_sec"] = df["환승소요시간"].apply(mmss_to_sec)

    use = df[["환승역명", "호선", "환승노선", "transfer_dist_m", "transfer_time_sec"]].copy()
    use = use.dropna(subset=["transfer_dist_m", "transfer_time_sec"])
    use["호선"] = use["호선"].astype(str).str.replace(" ", "", regex=False)

    # 역별 평균 난이도 점수(0~100)
    by_station = use.groupby("환승역명", as_index=False).agg(
        avg_dist_m=("transfer_dist_m", "mean"),
        avg_time_sec=("transfer_time_sec", "mean"),
        n=("환승노선", "count"),
    )

    by_station["dist_n"] = minmax(by_station["avg_dist_m"])
    by_station["time_n"] = minmax(by_station["avg_time_sec"])
    by_station["difficulty"] = ((0.5 * by_station["dist_n"]) + (0.5 * by_station["time_n"])) * 100
    by_station["difficulty"] = by_station["difficulty"].clip(0, 100)

    return use, by_station


# =========================
# 3) 역간(호선 내) 간선 만들기
#    segments.csv 컬럼:
#    ['연번','호선','역명','소요시간','역간거리(km)','호선별누계(km)']
# =========================
def load_segments_csv(path: str) -> pd.DataFrame:
    # 인코딩 자동 탐색 (서울 공공데이터는 cp949가 많음)
    enc_list = ["cp949", "euc-kr", "utf-8-sig", "utf-8", "latin1"]
    df = None
    last_err = None

    for enc in enc_list:
        try:
            df = pd.read_csv(path, encoding=enc, engine="python")
            break
        except Exception as e:
            last_err = e

    if df is None:
        raise RuntimeError(f"segments.csv를 읽지 못했습니다. 마지막 에러: {last_err}")

    df.columns = [c.strip() for c in df.columns]

    # 역명 공백 제거(동대문역사문화공원 같은 띄어쓰기 문제 방지)
    df["역명"] = df["역명"].astype(str).str.replace(r"\s+", "", regex=True)

    need = ["호선", "연번", "역명", "소요시간"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"segments.csv에 필요한 컬럼이 없습니다: {missing}\n현재 컬럼: {list(df.columns)}")

    df = df.sort_values(["호선", "연번"]).reset_index(drop=True)
    df["next_station"] = df.groupby("호선")["역명"].shift(-1)

    edges = df.dropna(subset=["next_station"]).copy()
    edges = edges.rename(
        columns={
            "호선": "line",
            "역명": "u_station",
            "next_station": "v_station",
            "소요시간": "time_raw",
        }
    )[
        ["line", "u_station", "v_station", "time_raw"]
    ]

    # 소요시간이 "02:00" 형태일 수도 있어서 분으로 변환
    def time_to_min(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if ":" in s:
            mm, ss = s.split(":")
            return int(mm) + int(ss) / 60.0
        return float(s)

    edges["time_min"] = edges["time_raw"].apply(time_to_min)
    edges["time_min"] = pd.to_numeric(edges["time_min"], errors="coerce").fillna(1.0)

    edges = edges.drop(columns=["time_raw"])

    # 양방향
    rev = edges.rename(columns={"u_station": "v_station", "v_station": "u_station"})
    edges = pd.concat([edges, rev], ignore_index=True)

    edges["line"] = edges["line"].astype(str).str.replace(" ", "", regex=False)

    return edges


# =========================
# 4) 그래프 구성 + 다익스트라
# =========================
def build_graph(segment_edges: pd.DataFrame, by_station: pd.DataFrame, alpha=0.03):
    """
    노드: '호선|역명'
    간선:
      - 호선 내 이동: 가중치 = time_min
      - 환승(같은 역명 다른 호선): 가중치 = (avg_time_sec/60) + alpha*difficulty
    """
    graph = {}

    def add_edge(a, b, w):
        graph.setdefault(a, []).append((b, float(w)))

    # 1) 호선 내 이동
    for _, r in segment_edges.iterrows():
        u = f"{r['line']}|{r['u_station']}"
        v = f"{r['line']}|{r['v_station']}"
        add_edge(u, v, r["time_min"])

    # 2) 환승 연결
    station_to_lines = (
        segment_edges.groupby("u_station")["line"]
        .apply(lambda x: sorted(set(x)))
        .to_dict()
    )

    info = by_station.set_index("환승역명")[["avg_time_sec", "difficulty"]].to_dict("index")

    for station, lines in station_to_lines.items():
        if len(lines) < 2:
            continue
        if station not in info:
            continue

        t_min = float(info[station]["avg_time_sec"]) / 60.0
        diff = float(info[station]["difficulty"])
        BASE_TRANSFER_MIN = 3.0          # 환승은 최소 3분은 먹는다고 가정(발표용으로 설득력 좋음)
        w_transfer = BASE_TRANSFER_MIN + (alpha * diff)

        nodes = [f"{ln}|{station}" for ln in lines]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                add_edge(nodes[i], nodes[j], w_transfer)
                add_edge(nodes[j], nodes[i], w_transfer)
    print("segments에 존재하는 환승가능역(2개 이상 노선) 개수:", sum(len(v)>=2 for v in station_to_lines.values()))
    print("실제 환승 간선 추가된 역 개수:", sum(1 for st, lines in station_to_lines.items() if len(lines)>=2 and st in info))


    return graph


def dijkstra(graph, start, goal):
    pq = [(0.0, start)]
    dist = {start: 0.0}
    prev = {start: None}

    while pq:
        cur_d, u = heapq.heappop(pq)
        if u == goal:
            break
        if cur_d != dist.get(u, np.inf):
            continue
        for v, w in graph.get(u, []):
            nd = cur_d + w
            if nd < dist.get(v, np.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if goal not in dist:
        return None, np.inf

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path, dist[goal]


def pick_nodes(segment_edges: pd.DataFrame, station: str):
    station = str(station).replace(" ", "")
    station_lines = (
        segment_edges.groupby("u_station")["line"]
        .apply(lambda x: sorted(set(x)))
        .to_dict()
    )
    lines = station_lines.get(station, [])
    return [f"{ln}|{station}" for ln in lines]


def best_path(graph, segment_edges, start_station, end_station):
    starts = pick_nodes(segment_edges, start_station)
    goals = pick_nodes(segment_edges, end_station)

    if not starts or not goals:
        raise ValueError("출발/도착역이 segments.csv에 없어요. 역명(띄어쓰기) 확인!")

    best = (None, np.inf)
    for s in starts:
        for g in goals:
            p, c = dijkstra(graph, s, g)
            if c < best[1]:
                best = (p, c)
    return best
def transfer_difficulty_sum(path, by_station):
    """
    path: ["4|서울역", ..., "4|사당", "2|사당", ...]
    by_station: 환승역별 difficulty 가진 DF
    return: (합계, 환승역 리스트)
    """
    if not path or by_station is None or by_station.empty:
        return 0.0, []

    diff_map = by_station.set_index("환승역명")["difficulty"].to_dict()

    transfer_stations = []
    prev_line, prev_station = path[0].split("|", 1)

    for node in path[1:]:
        line, station = node.split("|", 1)
        if station == prev_station and line != prev_line:
            transfer_stations.append(station)
        prev_line, prev_station = line, station

    s = sum(float(diff_map.get(st, 0.0)) for st in transfer_stations)
    return s, transfer_stations


def format_path_with_transfer_tag(path):
    """
    입력: ["4|서울역", "4|숙대입구", ..., "4|사당", "2|사당", ...]
    출력: "4|서울역 -> ... -> 4|사당 -> [환승]2|사당 -> ..."
    """
    if not path:
        return ""

    out = [path[0]]
    prev_line, prev_station = path[0].split("|", 1)

    for node in path[1:]:
        line, station = node.split("|", 1)

        # 같은 역에서 호선이 바뀌면 환승으로 표시
        if station == prev_station and line != prev_line:
            out.append(f"[환승]{node}")
        else:
            out.append(node)

        prev_line, prev_station = line, station

    return " -> ".join(out)



# =========================
# 5) 시각화 3종
# =========================
def plot_top_transfer_stations(by_station, top_n=20):
    top = by_station.sort_values("difficulty", ascending=False).head(top_n)
    plt.figure()
    plt.barh(top["환승역명"][::-1], top["difficulty"][::-1])
    plt.xlabel("환승 난이도 점수 (0~100)")
    plt.title(f"환승 난이도 TOP {top_n} (역별 평균)")
    plt.tight_layout()
    plt.show()


def plot_difficulty_distribution(by_station):
    plt.figure()
    plt.hist(by_station["difficulty"].dropna(), bins=20)
    plt.xlabel("환승 난이도 점수 (0~100)")
    plt.ylabel("역 개수")
    plt.title("환승 난이도 분포")
    plt.tight_layout()
    plt.show()


def plot_dist_time_scatter(by_station):
    plt.figure()
    plt.scatter(by_station["avg_dist_m"], by_station["avg_time_sec"])
    plt.xlabel("평균 환승거리 (m)")
    plt.ylabel("평균 환승시간 (sec)")
    plt.title("환승거리 vs 환승시간 (역별 평균)")
    plt.tight_layout()
    plt.show()

def edge_weight(graph, a, b):
    # graph[a] 안에서 b로 가는 가중치 찾기
    for nxt, w in graph.get(a, []):
        if nxt == b:
            return float(w)
    return 0.0  # 혹시 못 찾으면 0으로

def plot_route_cumulative_cost(graph_fast, graph_easy, path_fast, path_easy):
    import numpy as np
    import matplotlib.pyplot as plt

    def step_costs(graph, path):
        costs = []
        for i in range(len(path) - 1):
            costs.append(edge_weight(graph, path[i], path[i + 1]))
        return np.array(costs, dtype=float)

    c_fast = step_costs(graph_fast, path_fast)
    c_easy = step_costs(graph_easy, path_easy)

    cum_fast = np.insert(np.cumsum(c_fast), 0, 0.0)
    cum_easy = np.insert(np.cumsum(c_easy), 0, 0.0)


    plt.figure()
    plt.plot(cum_fast, label="최단시간")
    plt.plot(cum_easy, label="환승쉬움(피로도 반영)")
    plt.xlabel("이동 단계(step)")
    plt.ylabel("누적 비용(분)")
    plt.title("경로별 누적 비용 비교")
    plt.legend()
    plt.tight_layout()
    plt.show()



# =========================
# 6) 실행
# =========================
def normalize_station(s: str) -> str:
    return str(s).strip().replace(" ", "")

if __name__ == "__main__":
    # 1) 데이터 준비(한 번만 로딩)
    df_transfer_raw = fetch_all_transfer(per_page=100)
    transfer_rows, by_station = clean_transfer(df_transfer_raw)
    seg_edges = load_segments_csv(SEGMENTS_CSV)

    g_easy = build_graph(seg_edges, by_station, alpha=ALPHA)
    g_fast = build_graph(seg_edges, by_station, alpha=0.0)
    
        # ---- 그래프(데이터 전체 요약) 1회 표시 ----
    plot_top_transfer_stations(by_station, top_n=20)
    plot_difficulty_distribution(by_station)
    plot_dist_time_scatter(by_station)


    # ✅ 역 목록(입력 확인용)
    station_set = sorted(set(seg_edges["u_station"].astype(str)))

    print("\n역 입력 예시: 서울역, 강남, 고속터미널, 홍대입구")
    print("종료하려면 exit 입력\n")

    while True:
        start = input("출발역을 입력하세요: ").strip()
        if start.lower() == "exit":
            break
        end = input("도착역을 입력하세요: ").strip()
        if end.lower() == "exit":
            break

        start_n = normalize_station(start)
        end_n = normalize_station(end)

        try:
            p1, c1 = best_path(g_fast, seg_edges, start_n, end_n)
            p2, c2 = best_path(g_easy, seg_edges, start_n, end_n)

            print("\n[최단시간 경로]")
            print(format_path_with_transfer_tag(p1))
            print(f"총 비용(분): {c1:.1f}")

            print("\n[환승이 쉬운 경로(피로도 반영)]")
            print(format_path_with_transfer_tag(p2))
            print(f"총 비용(분): {c2:.1f}")

            print(f"\n시간 증가(대략): {c2 - c1:.1f}분  (ALPHA={ALPHA})\n")
            
            sum1, ts1 = transfer_difficulty_sum(p1, by_station)
            sum2, ts2 = transfer_difficulty_sum(p2, by_station)

            print(f"환승역: {ts1} / 환승난이도 합: {sum1:.1f}")
            print(f"환승역: {ts2} / 환승난이도 합: {sum2:.1f}")


        except ValueError:
            # ✅ 역명 입력이 틀렸을 때: 비슷한 후보 몇 개 추천
            def suggest(query, top=10):
                q = normalize_station(query)
                cand = [st for st in station_set if q in st]
                return cand[:top]

            print("\n⚠️ 역명 찾기를 실패하였습니다.")
            print("출발역 후보:", suggest(start))
            print("도착역 후보:", suggest(end))
            print("※ 역명 띄어쓰기/오타를 확인해주세요.\n")
