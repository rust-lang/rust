#![allow(non_camel_case_types)]

type point = (isize, isize);

fn f(p: point, x: isize, y: isize) {
    let (a, b) = p;
    assert_eq!(a, x);
    assert_eq!(b, y);
}

pub fn main() {
    let p: point = (10, 20);
    let (a, b) = p;
    assert_eq!(a, 10);
    assert_eq!(b, 20);
    let p2: point = p;
    f(p, 10, 20);
    f(p2, 10, 20);
}
