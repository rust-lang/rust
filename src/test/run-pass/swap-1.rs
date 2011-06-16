fn main() {
    auto x = 3;
    auto y = 7;
    x <-> y;
    assert (x == 7);
    assert (y == 3);
}
