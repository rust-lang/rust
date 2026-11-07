// A regression test for https://github.com/rust-lang/rust/issues/148121

pub trait Super<X> {
    type X;
}

pub trait Zelf<X>: Super<X> {}

pub trait A {}

impl A for dyn Super<X = ()> {}
//~^ ERROR: trait takes 1 generic argument but 0 generic arguments were supplied

impl A for dyn Zelf<X = ()> {}
//~^ ERROR: trait takes 1 generic argument but 0 generic arguments were supplied

fn main() {}
