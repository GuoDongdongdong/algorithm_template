#include<bits/stdc++.h>
using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

template<typename T>
class SegTreeLazySumAdd{
    
    const vector<T>& arr;
    vector<T> tree;
    vector<T> lazy;

    void __build(int p, int l, int r) {
        if(l == r) {
            tree[p] = arr[l];
            return ;
        }
        __build(2 * p, l, (l + r) / 2);
        __build(2 * p + 1, (l + r) / 2 + 1, r);
        tree[p] = tree[2 * p] + tree[2 * p + 1];
    }

    void update(int p, int cl, int cr) {
        int cm = (cl + cr) / 2;
        if(cl != cr && lazy[p]) {
            lazy[2 * p] += lazy[p];
            lazy[2 * p + 1] += lazy[p];
            tree[2 * p] += lazy[p] * (cm - cl + 1);
            tree[2 * p + 1] += lazy[p] * (cr - cm);
            lazy[p] = 0;
        }
    }

    T __range_sum(int p, int l, int r, int cl, int cr) {
        if(l <= cl && r >= cr) return tree[p];
        update(p, cl, cr);
        int cm = (cl + cr) / 2;
        T re = 0;
        if(l <= cm) {
            re += __range_sum(2 * p, l, r, cl, cm);
        }
        if(r > cm) {
            re += __range_sum(2 * p + 1, l, r, cm + 1, cr);
        }
        return re;
    }

    void __range_add(int p, int l, int r, int cl, int cr, T val) {
        if(l <= cl && r >= cr) {
            lazy[p] += val;
            tree[p] += val * (cr - cl + 1);
            return ;
        }
        update(p, cl, cr);
        int cm = (cl + cr) / 2;
        if(l <= cl) {
            __range_add(2 * p, l, r, cl, cm, val);
        }
        if(r > cm) {
            __range_add(2 * p + 1, l, r, cm + 1, cr, val);
        }
        tree[p] = tree[2 * p] + tree[2 * p + 1];
    }

public:
    explicit SegTreeLazySumAdd(const vector<T>& _arr): arr(_arr), tree(vector<T>(4 * arr.size())), lazy(vector<T>(tree.size())) {
        __build(1, 0, arr.size() - 1);
    }

    T range_sum(int l, int r) {
        return __range_sum(1, l, r, 0, arr.size() - 1);
    }

    void range_add(int l, int r, T val) {
        __range_add(1, l, r, 0, arr.size() - 1, val);
    }
};

class Tree { // 树上算法 
    vector<vector<int> > g; // 节点边的关系
    vector<vector<int> > fa; // fa[i, j] 代表 节点i到根节点路径上第2^j节点, -1表示不存在
    vector<int> deep; // 节点深度

public:
    Tree(const vector<vector<int> >& edges) {// 描述树中边的关系 边的数量等于节点数 - 1
        int n = edges.size() + 1;
        int m = log2(n) + 1;
        g.resize(n);
        fa.resize(n, vector<int>(m, -1));
        deep.resize(n, 0);
        for(auto& e: edges) {
            int u = e[0], v = e[1];
            g[u].push_back(v);
            g[v].push_back(u);
        }
        function<void(int, int)> dfs = [&](int u, int father){ // 得到节点深度和父节点信息
            fa[u][0] = father;
            for(auto v: g[u]) {
                if(v == father) continue;
                deep[v] = deep[u] + 1;
                dfs(v, u);
            }
        };
        dfs(0, -1);
        for(int i = 1; i < m; ++i) {
            for(int j = 0; j < n; ++j) {
                if(fa[j][i - 1] != -1) {
                    fa[j][i] = fa[fa[j][i - 1]][i - 1];
                }
            }
        }
    }
    
    int get_k_ancestor(int x, int k) { // 节点x的第k个祖先
        for(int i = 1, j = 0; i <= k && x != -1; i <<= 1, ++j) {
            if(i & k) {
                x = fa[x][j];
            }
        }
        return x;
    }

    int get_lca(int x, int y) { // 节点x y的最近公共祖先
        if(deep[x] > deep[y]) swap(x, y);
        y = get_k_ancestor(y, deep[y] - deep[x]);
        if(x == y) return x;
        for(int i = fa[x].size() - 1; i >= 0; --i) {
            int nx = fa[x][i], ny = fa[y][i];
            if(nx != ny) {
                x = nx;
                y = ny;
            } 
        }
        return fa[x][0];
    }
};

class UnionCheckSet { // 并查集
    vector<int> fa, size; // size代表每个节点为根的子树大小 用于启发式合并

public:
    UnionCheckSet(int n): fa(n), size(n, 1) { 
        iota(fa.begin(), fa.end(), 0);
    }

    int find(int x) { // 路径压缩
        // return fa[x] == x ? x : fa[x] = find(fa[x]);
        if(fa[x] != x) {
            fa[x] = find(fa[x]);
        }
        return fa[x];
    }

    void unite(int x, int y) { // 启发式合并，将节点少的树合并到节点多的树
        int rootx = find(x), rooty = find(y);
        if(x == y) return ;
        if(size[x] > size[y]) {
            swap(x, y);
        }
        fa[rootx] = rooty;
        size[rooty] += size[rootx];
    }
};

class ChainForwardStar { // 链式前向星
    int cnt; // 边的序号 从0开始
    int n; // 顶点个数
    vector<int> next; // 同一个节点的边下一个邻接边
    vector<int> head; // 节点的第一个边
    vector<pair<int, int> > to; // 边的下一个节点和权
    
public:
    ChainForwardStar(int _n, int m): cnt(0), n(_n), next(m), head(n, -1), to(m){}

    void add_edge(int u, int v, int w = 0) {
        next.push_back(head[u]);
        head[u] = next.size() - 1;
        to.push_back(make_pair(v, w));
    }
    
    void dfs(int s) {
        vector<bool> vis(n, false);
        function<void(int)> _dfs = [&](int s){
            if(vis[s]) return ;
            vis[s] = true;
            for(int e = head[s]; e != -1; e = next[e]) {
                _dfs(to[e].first);
            }
        };
        _dfs(s);
    }

};

class Floyd { // 所有点之间最短路，不能有负环 O(n³)
    const int inf = 0x3f3f3f3f;
    int n;
    vector<vector<vector<int> > > dp;
public:
    Floyd(int n, const vector<vector<int> >& g): n(n), dp(n + 1, vector<vector<int> >(n + 1, vector<int>(n + 1, inf))){ // 顶点为1到n
        for(int i = 1; i <= n; ++i) {
            dp[0][i][i] = 0;
        }
        for(auto& e: g) {
            int u = e[0], v = e[1], c = e[2];
            dp[0][u][v] = c;
            dp[0][v][u] = c; // 无向图
        }
        for(int k = 1; k <= n; ++k) {
            for(int u = 1; u <= n; ++u) {
                for(int v = 1; v <= n; ++v) {
                    dp[k][u][v] = min(dp[k - 1][u][v], dp[k - 1][u][k] + dp[k - 1][k][v]);
                }
            }
        }
    }
    int query(int s, int t) {
        return dp[n][s][t] == inf ? -1 : dp[n][s][t];
    }
};

class BellMan_Ford { // 可以判定负环 O(V * max(V, E))
    const int inf = 0x3f3f3f3f;
    int n;
    vector<int> dist;
    vector<vector<pair<int, int> > > g;
public:
    BellMan_Ford(int _n, const vector<vector<int> >& edges): n(_n), dist(n, inf), g(n) {
        for(auto &e : edges) { // 顶点为0到n - 1
            int u = e[0], v = e[1], c = e[2];
            g[u].emplace_back(v, c);
            g[v].emplace_back(u, c); // 无向图
        }
    }

    bool init(int s) { // 源点s
        dist[s] = 0;
        bool flag; // 是否有松弛操作
        for(int i = 1; i <= n; ++i) { // n - 1轮后应该没有可以松弛的边
            flag = false;
            for(int u = 0; u < n; ++u) {
                if(dist[u] == inf) continue;
                for(auto [v, w] : g[u]) {
                    if(dist[v] > dist[u] + w) {
                        dist[v] = dist[u] + w;
                        flag = true;
                    }
                }
            }
            if(!flag) break;
        }
        if(flag) { // 有负环可以无限松弛
            return true;
        }
        return false;
    }

    int query(int t) {
        return dist[t];
    }
};

class SPFA { // 可以判定负环 最坏复杂度和BellMan Ford一致
    const int inf = 0x3f3f3f3f;
    int n;
    vector<vector<pair<int, int> > > g;
    vector<int> dist;

public:
    SPFA(int _n, const vector<vector<int> >& grid) :n(_n), g(n), dist(n, inf) {
        for(auto& e : grid) { // 顶点为0到n - 1
            int u = e[0], v = e[1], c = e[2];
            g[u].emplace_back(v, c);
            g[v].emplace_back(u, c); // 无向图
        }
    }
    
    bool init(int s) {
        dist[s] = 0;
        vector<bool> vis(n, false); // 保证每个节点在队列仅一次 否则无法保证时间复杂度
        vector<int> cnt(n, 0); // 记录源点到节点最短路的边数
        queue<int> q;
        vis[s] = true;
        q.push(s);
        while(!q.empty()) {
            int u = q.front(); q.pop();
            vis[u] = false;
            for(auto& [v, w] : g[u]) {
                if(dist[v] > dist[u] + w) {
                    dist[v] = dist[u] + w;
                    cnt[v] = cnt[u] + 1;
                    if(cnt[v] >= n) return true; // 最短路径大于等于n说明存在负环
                    if(!vis[v]) {
                        q.push(v);
                        vis[v] = true;
                    }
                }
            }
        }
        return false;
    }
    
    int query(int t) {
        return dist[t] == inf ? -1 : dist[t];
    }
};

class Dijkstra { // 不能有负边 堆优化 O(E * log(E)) = O(E * log(V))
    // 稠密图时 O(V² * log(V)) 比朴素的Dijkstra算法O(V²)差
    // 如果是有向无环图DAG可以用拓扑排序每次松弛入度为0的边，O(V + E)
    const int inf = 0x3f3f3f3f;
    int n;
    vector<int> dist;
    vector<vector<pair<int, int> > > g;
    struct node {
        int u, dist;
        node(int _u, int _dist):u(_u), dist(_dist) {}

        bool operator>(const node& other) const noexcept {
            return dist > other.dist;
        }
    };
public:
    Dijkstra(int _n, int s, const vector<vector<int> >& grid):n(_n), dist(n, inf), g(n) {
        for(auto& e: grid) { // 顶点从0到n-1
            int u = e[0], v = e[1], c = e[2];
            g[u].emplace_back(v, c);
            g[v].emplace_back(u, c); // 无向图
        }
        dist[s] = 0;
        vector<bool> vis(n, false);
        priority_queue<node,vector<node>, greater<node> > pq;
        pq.emplace(s, 0);
        while(!pq.empty()) {
            auto [u, d] = pq.top(); pq.pop();
            vis[u] = true;
            for(auto [v, w] : g[u]) {
                if(!vis[v] && dist[v] > dist[u] + w) { 
                    dist[v] = dist[u] + w;
                    pq.emplace(v, dist[v]);
                }
            }
        }
        /*
        暴力枚举O(n²)
        for(int i = 1; i <= n; ++i) { // 每次循环找到一个距离源点最近的且未访问过的节点，一共n个
            int u = -1;
            int Min = inf;
            for(int i = 0; i < n; ++i) {
                if(dist[i] < Min && !vis[i]) {
                    u = i;
                    Min = dist[i];
                }
            }
            vis[u] = true;
            for(auto [v, w] : g[u]) {
                if(dist[v] > dist[u] + w) {
                    dist[v] = dist[u] + w;
                }
            }
        }
        */
    }

    int query(int t) {
        return dist[t] == inf ? -1 : dist[t];
    }
};

class EdmondsKarp { // 网络最大流 O(VE²)
    struct edge {
        int u, v, cap, flow;
        edge(int _u, int _v, int _cap, int _flow): u(_u), v(_v), cap(_cap), flow(_flow){}
    };
    const int inf = 0x3f3f3f3f;
    int n; // 顶点个数
    vector<edge> edges; // 所有边 包括反向边
    vector<vector<int> > g; // 顶点v的所有边在edges中的下标
public:
    EdmondsKarp(int _n, const vector<vector<int> >& _edges): n(_n), edges(), g(_n) {
        for(auto& e: _edges) {
            int u = e[0], v = e[1], c = e[2];
            edges.emplace_back(u, v, c, 0);
            edges.emplace_back(v, u, 0, 0); // 反向边
            g[u].push_back(edges.size() - 2);
            g[v].push_back(edges.size() - 1);
        }
    }
    int maxflow(int s, int t) { // 源点s到汇点t的最大流
        int re = 0;
        while(true) {
            vector<int> flows(n, 0); // 每个点获得的流量
            vector<int> pe(n, -1); // 每个点v在bfs路径上u到v的边的下标
            queue<int> q;
            q.push(s);
            flows[s] = inf;
            while(!q.empty()) {
                int u = q.front(); q.pop();
                for(auto e : g[u]) {
                    int v = edges[e].v, cap = edges[e].cap, flow = edges[e].flow;
                    if(!flows[v] && cap - flow > 0) {
                        pe[v] = e;
                        flows[v] = min(flows[u], cap - flow);
                        q.push(v);
                    }
                }
                if(flows[t]) break;
            }
            if(!flows[t]) { // s和t不在一个连通分量
                break;
            }
            for(int v = t; v != s; v = edges[pe[v]].u) {
                edges[pe[v]].flow += flows[t];
                edges[pe[v] ^ 1].flow -= flows[t];
            }
            re += flows[t];
        }
        return re;
    }
};

class SSP { 
// 最小费用最大流 贪心的求出s到t的花费最小的路径(spfa，不能用dijkstra，因为存在负花费的边)，然后在这条路径上增加流量，直至不存在增广路
    struct edge {
        int next, v, flow, cap, cost;
        edge(int _next, int _v, int _flow, int _cap, int _cost): next(_next), v(_v), flow(_flow), cap(_cap), cost(_cost) {}
    };
    int n; // 节点个数
    vector<edge> edges; // 链式前向星
    vector<int> head;
    const int inf = 0x3f3f3f3f;

    void add_edge(int u, int v, int cap, int cost) {
        edges.emplace_back(head[u], v, 0, cap, cost);
        head[u] = edges.size() - 1;
    }

    bool spfa(int s, int t) {
        queue<int > q;
        vector<int> vis(n, false);
        vector<int> dist(n, inf);
        vector<int> pre(n, -1);
        vector<int> flows(n, 0);
        q.push(s);
        dist[s] = 0;
        flows[s] = inf;
        while(!q.empty()) {
            int u = q.front(); q.pop();
            vis[u] = false;
            for(int e = head[u]; e != -1; e = edges[e].next) {
                auto [_, v, flow, cap, cost] = edges[e];
                if(cap - flow > 0 && dist[v] > dist[u] + cost) {
                    dist[v] = dist[u] + cost;
                    flows[v] = min(cap - flow, flows[u]);
                    pre[v] = e;
                    if(!vis[v]){
                        q.push(v);
                        vis[v] = true;
                    }
                }
            }
        }
        if(!flows[t]) return false;
        static int debug = 0;
        cout << debug++ << ": ";
        for(int u = t; u != s; u = edges[pre[u] ^ 1].v) {
            cout << u << "," << edges[pre[u]].cost << " <- ";
            edges[pre[u]].flow += flows[t];
            edges[pre[u] ^ 1].flow -= flows[t];
            mincost += edges[pre[u]].cost * flows[t];
        }
        cout << s << endl;
        maxflow += flows[t];
        return true;
    }

    void solve(int s, int t) {
        while(spfa(s, t));
    }

public:
    int maxflow, mincost; // 最大流和最小花费
    SSP(int _n, int s, int t, const vector<vector<int> >& g): n(_n), maxflow(0), mincost(0), head(n, -1), edges() {
        for(auto& e : g) {
            // 点u到点v，管道容量为cap，单位花费为cost
            int u = e[0], v = e[1], cap = e[2], cost = e[3];
            add_edge(u, v, cap, cost);
            add_edge(v, u, 0, -cost);
        }
        solve(s, t);
    }

};

template<typename T, int N>
class Matrix { // n阶方阵
    vector<vector<T> > matrix;

public:
    Matrix(const vector<vector<T> >& _m): matrix(_m){}

    Matrix(): matrix(N, vector<T>(N, 0)) {}

    void unit() { // 更新为单位阵
        for(int i = 0; i < N; ++i) {
            for(int j = 0; j < N; ++j) {
                matrix[i][j] = i == j ? 1 : 0;
            }
        }
    }

    void mat(const Matrix& b) {
        Matrix tmp;
        for(int i = 0; i < N; ++i) {
            for(int j = 0; j < N; ++j) {
                for(int k = 0; k < N; ++k) {
                   tmp.matrix[i][j] += matrix[i][k] * b.matrix[k][j];
                }
            }
        }
        *this = tmp;
    }

    void quick_pow(int n) {
        Matrix tmp;
        tmp.unit();
        while(n) {
            if(n & 1) {
                tmp.mat(*this);
            }
            mat(*this);
            n >>= 1;
        }
        *this = tmp;
    }

    T get_ij(int i, int j) {
        return matrix[i][j];
    }
};

class StringMatching { // 字符串匹配相关算法
    vector<int> prefix_function(const string& s) { // O(n)计算一个字符串的最长真前后缀相同长度
        vector<int> pre(s.size(), 0);
        for(int i = 1; i < s.size(); ++i) {
            int j = pre[i - 1];
            while(j && s[j] != s[i]) {
                j = pre[j - 1];
            }
            if(s[j] == s[i]) ++j;
            pre[i] = j;
        }
        return pre;
    }

public:
    vector<int> kmp(const string& s, const string& t) { // O(len(s) + len(t)) 求出串t在串s中出现的下标（允许重叠）
        vector<int> pre = prefix_function(t);
        vector<int> res;
        int index_t = 0;
        for(int i = 0; i < s.size(); ++i) {
            while(index_t && t[index_t] != s[i]) {
                index_t = pre[index_t - 1];
            }
            if(t[index_t] == s[i]) {
                ++index_t;
            }
            if(index_t == t.size()) {
                res.push_back(i - index_t + 1);
                index_t = pre[index_t - 1];
            }
        }
        return res;
    }
};