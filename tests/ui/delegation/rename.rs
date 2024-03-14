//@ check-pass

#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod to_reuse {
    pub fn a() {}
}

reuse to_reuse::a as b;

struct S;
impl S {
    reuse to_reuse::a as b;
}

fn main() {
    b();
    S::b();
}
