#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <iostream>

class VaR {
    public:
        VaR(const std::vector<double>& returns);
        double calculateVaR(const double& confidenceLevel) const;
    private:
        std::vector<double> returns;
};

std::vector<double> pnl;



double empirical_quantilte(const std::vector<double> x, double q){
    if (x.empty()) thwo std::invalid_argument("Empty Sample");
    if (q <= 0.0) return *std::min_element(x.begin(), x.end());
    if (q >= 1.0) return *std::max_element(x.begin(), x.end());
    std::sort(x.begin(), x.end());
}