// xfail-fast - check-fast doesn't understand aux-build
// aux-build:trait_inheritance_overloading_xc.rs

extern mod trait_inheritance_overloading_xc;
use trait_inheritance_overloading_xc::{MyNum, MyInt};

fn f<T:Copy MyNum>(x: T, y: T) -> (T, T, T) {
    return (x + y, x - y, x * y);
}

pure fn mi(v: int) -> MyInt { MyInt { val: v } }

fn main() {
    let (x, y) = (mi(3), mi(5));
    let (a, b, c) = f(x, y);
    assert a == mi(8);
    assert b == mi(-2);
    assert c == mi(15);
}

