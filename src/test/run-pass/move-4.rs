use std;
import std::uint;

fn test(@tup(int, int, int) foo) -> @tup(int, int, int) {
    auto bar <- foo;
    auto baz <- bar;
    auto quux <- baz;
    ret quux;
}

fn main() {
    auto x = @tup(1,2,3);
    auto y = test(x);
    assert (y._2 == 3);
}
