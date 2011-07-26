fn main() {
    auto sum = 0;
    for (auto x in ~[1, 2, 3, 4, 5]) {
        sum += x;
    }
    assert sum == 15;
}
