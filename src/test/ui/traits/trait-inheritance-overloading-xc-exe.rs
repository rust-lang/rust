// run-pass
// aux-build:trait_inheritance_overloading_xc.rs


extern crate trait_inheritance_overloading_xc;
use trait_inheritance_overloading_xc::{MyNum, MyInt};

fn f<T:MyNum>(x: T, y: T) -> (T, T, T) {
    return (x.clone() + y.clone(), x.clone() - y.clone(), x * y);
}

fn mi(v: isize) -> MyInt { MyInt { val: v } }

pub fn main() {
    let (x, y) = (mi(3), mi(5));
    let (a, b, c) = f(x, y);
    assert_eq!(a, mi(8));
    assert_eq!(b, mi(-2));
    assert_eq!(c, mi(15));
}
