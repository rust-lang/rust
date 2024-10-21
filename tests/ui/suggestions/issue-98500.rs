//@ aux-build:dyn-incompatible.rs

extern crate dyn_incompatible;

pub trait B where
    Self: dyn_incompatible::A,
{
    fn f2(&self);
}

struct S(Box<dyn B>);
//~^ ERROR the trait `B` cannot be made into an object

fn main() {}
