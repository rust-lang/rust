//@ run-pass
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use std::cmp;

#[derive(Debug)]
struct foo { a: isize, b: isize, c: isize }

impl cmp::PartialEq for foo {
    fn eq(&self, other: &foo) -> bool {
        (*self).a == (*other).a &&
        (*self).b == (*other).b &&
        (*self).c == (*other).c
    }
    fn ne(&self, other: &foo) -> bool { !(*self).eq(other) }
}

const x : foo = foo { a:1, b:2, c: 3 };
const y : foo = foo { b:2, c:3, a: 1 };
const z : &'static foo = &foo { a: 10, b: 22, c: 12 };
const w : foo = foo { a:5, ..x };

pub fn main() {
    assert_eq!(x.b, 2);
    assert_eq!(x, y);
    assert_eq!(z.b, 22);
    assert_eq!(w.a, 5);
    assert_eq!(w.c, 3);
    println!("{:#x}", x.b);
    println!("{:#x}", z.c);
}
