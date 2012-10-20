// xfail-fast - check-fast doesn't understand aux-build
// aux-build:trait_inheritance_overloading_xc.rs

extern mod trait_inheritance_overloading_xc;
use trait_inheritance_overloading_xc::MyNum;

fn f<T:Copy MyNum>(x: T, y: T) -> (T, T, T) {
    return (x + y, x - y, x * y);
}

fn main() {
    let (x, y) = (3, 5);
    let (a, b, c) = f(x, y);
    assert a == 8;
    assert b == -2;
    assert c == 15;
}


