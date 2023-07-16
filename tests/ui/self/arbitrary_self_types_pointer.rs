#![feature(arbitrary_self_types)]

struct A;

impl A {
    fn m(self: *const Self) {}
    //~^ ERROR: `*const A` cannot be used as the type of `self` without the `arbitrary_self_types_pointers` feature
}

trait B {
    fn bm(self: *const Self) {}
    //~^ ERROR: `*const Self` cannot be used as the type of `self` without the `arbitrary_self_types_pointers` feature
}

fn main() {
    let a = A;
    let ptr = &a as *const A;
    ptr.m();
}
