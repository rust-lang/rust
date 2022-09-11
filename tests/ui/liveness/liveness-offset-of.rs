#![feature(offset_of)]
#![deny(dead_code)]

use std::mem::offset_of;

struct Alpha {
    a: (),
    b: (), //~ ERROR field `b` is never read
    c: Beta,
}

struct Beta {
    a: (), //~ ERROR field `a` is never read
    b: (),
}

struct Gamma {
    a: (), //~ ERROR field `a` is never read
    b: (),
}

fn main() {
    offset_of!(Alpha, a);
    offset_of!(Alpha, c.b);
    offset_of!((Gamma,), 0.b);
}
