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

struct Delta {
    a: (),
    b: (), //~ ERROR field `b` is never read
}

trait Trait {
    type Assoc;
}
impl Trait for () {
    type Assoc = Delta;
}

struct Project<T: Trait> {
    a: u8, //~ ERROR field `a` is never read
    b: <T as Trait>::Assoc,
}

fn main() {
    offset_of!(Alpha, a);
    offset_of!(Alpha, c.b);
    offset_of!((Gamma,), 0.b);
    offset_of!(Project::<()>, b.a);
}
