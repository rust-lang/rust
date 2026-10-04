//@ aux-build:dyn-incompatible.rs

extern crate dyn_incompatible;

pub trait B where
    Self: dyn_incompatible::A,
{
    fn f2(&self);
}

struct S(Box<dyn B>);
//~^ ERROR the trait `B` is not dyn compatible

fn main() {}
