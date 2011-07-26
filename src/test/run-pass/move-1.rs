fn test(bool x, @rec(int x, int y, int z) foo) -> int {
    auto bar = foo;
    let @rec(int x,int y, int z) y;
    if (x) {
        y <- bar;
    } else {
        y = @rec(x=4, y=5, z=6);
    }
    ret y.y;
}

fn main() {
    auto x = @rec(x=1, y=2, z=3);
    assert (test(true, x) == 2);
    assert (test(true, x) == 2);
    assert (test(true, x) == 2);
    assert (test(false, x) == 5);
}
