#![feature(arbitrary_self_types)]

struct A;

impl A {
    fn m(self: std::ptr::NonNull<Self>) {}
    //~^ ERROR: invalid `self` parameter type
    fn n(self: &std::ptr::NonNull<Self>) {}
    //~^ ERROR: invalid `self` parameter type
}

fn main() {
}
