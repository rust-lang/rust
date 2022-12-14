// run-pass
#![allow(dead_code)]
// Issue #1112
// Alignment of interior pointers to dynamic-size types


struct X<T> {
    a: T,
    b: u8,
    c: bool,
    d: u8,
    e: u16,
    f: u8,
    g: u8
}

pub fn main() {
    let x: X<isize> = X {
        a: 12345678,
        b: 9,
        c: true,
        d: 10,
        e: 11,
        f: 12,
        g: 13
    };
    bar(x);
}

fn bar<T>(x: X<T>) {
    assert_eq!(x.b, 9);
    assert_eq!(x.c, true);
    assert_eq!(x.d, 10);
    assert_eq!(x.e, 11);
    assert_eq!(x.f, 12);
    assert_eq!(x.g, 13);
}
