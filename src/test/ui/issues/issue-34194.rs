// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]

struct A {
    a: &'static (),
}

static B: &'static A = &A { a: &() };
static C: &'static A = &B;

fn main() {}
