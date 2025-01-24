#![feature(arbitrary_self_types)]

struct A;

impl A {
    fn m(self: std::rc::Weak<Self>) {}
    //~^ ERROR: invalid `self` parameter type
    fn n(self: std::sync::Weak<Self>) {}
    //~^ ERROR: invalid `self` parameter type
}

fn main() {
}
