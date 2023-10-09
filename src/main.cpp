#include<bits/stdc++.h>
using namespace std;

int main() {
    freopen("in.txt", "r", stdin);
    freopen("out.txt", "w", stdout);
    int n; cin >> n;
    vector<int> v(n);
    for(auto& e : v) cin >> e;
    if(v.size() == 2) {
        cout << max(v[0], v[1]) << endl;
        return 0;
    }
    vector<vector<int> > dp(n, vector<int>(2, 0));
    for(int i = 1; i < n; ++i) {
        dp[i][1] = max(dp[i - 1][0] + v[i], dp[i - 1][1]);
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1]);
    }
    int ans = max(dp[n - 1][0], dp[n - 1][1]);
    dp[0][1] = v[0];
    for(int i = 1; i < n; ++i) {
        dp[i][1] = max(dp[i - 1][0] + v[i], dp[i - 1][1]);
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1]);
    }
    ans = max(ans, dp[n - 1][0]);
    cout << ans;
}
