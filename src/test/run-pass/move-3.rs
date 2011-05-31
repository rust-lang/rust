use std;
import std::uint;

fn test(bool x, @tup(int, int, int) foo) -> int {
    auto bar = foo;
    let @tup(int,int,int) y;
    if (x) {
        y <- bar;
    } else {
        y = @tup(4,5,6);
    }
    ret y._1;
}

fn main() {
    auto x = @tup(1,2,3);
    for each (uint i in uint::range(0u, 10000u)) {
        assert (test(true, x) == 2);
    }
    assert (test(false, x) == 5);
}
