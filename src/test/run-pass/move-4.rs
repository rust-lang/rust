
use std;
import std::uint;

fn test(@rec(int a, int b, int c) foo) -> @rec(int a, int b, int c) {
    auto bar <- foo;
    auto baz <- bar;
    auto quux <- baz;
    ret quux;
}

fn main() {
    auto x = @rec(a=1, b=2, c=3);
    auto y = test(x);
    assert (y.c == 3);
}