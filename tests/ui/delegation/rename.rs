//@ check-pass

#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod to_reuse {
    pub fn a() {}
    pub fn b() {}
}

reuse to_reuse::a as x;
reuse to_reuse::{a as y, b as z};

struct S;
impl S {
    reuse to_reuse::a as x;
    reuse to_reuse::{a as y, b as z};
}

fn main() {
    x();
    y();
    z();
    S::x();
    S::y();
    S::z();
}
