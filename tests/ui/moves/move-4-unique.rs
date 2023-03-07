// run-pass
#![allow(dead_code)]

struct Triple {a: isize, b: isize, c: isize}

fn test(foo: Box<Triple>) -> Box<Triple> {
    let foo = foo;
    let bar = foo;
    let baz = bar;
    let quux = baz;
    return quux;
}

pub fn main() {
    let x = Box::new(Triple{a: 1, b: 2, c: 3});
    let y = test(x);
    assert_eq!(y.c, 3);
}
