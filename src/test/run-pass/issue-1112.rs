// Issue #1112
// Alignment of interior pointers to dynamic-size types

use std;
import std::ptr::addr_of;

type x<T> = {
    a: T,
    b: u8,
    c: bool,
    d: u8,
    e: u16,
    f: u8,
    g: u8
};

fn main() {
    let x: x<int> = {
        a: 12345678,
        b: 9u8,
        c: true,
        d: 10u8,
        e: 11u16,
        f: 12u8,
        g: 13u8
    };
    bar(x);
}

fn bar<T>(x: x<T>) {
    assert x.b == 9u8;
    assert x.c == true;
    assert x.d == 10u8;
    assert x.e == 11u16;
    assert x.f == 12u8;
    assert x.g == 13u8;
}