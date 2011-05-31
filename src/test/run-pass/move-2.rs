fn main() {
    auto x = @tup(1,2,3);
    auto y <- x;
    assert (y._1 == 2);
}
