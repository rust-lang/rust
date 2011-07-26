

fn main() {
    auto x = @rec(x=1, y=2, z=3);
    auto y <- x;
    assert (y.y == 2);
}