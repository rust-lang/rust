fn main() {
    auto x = ~[10, 20, 30];
    auto sum = 0;
    for (auto x in x) { sum += x; }
    assert sum == 60;
}
