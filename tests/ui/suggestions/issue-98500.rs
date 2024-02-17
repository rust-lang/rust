//@ aux-build:not-object-safe.rs

extern crate not_object_safe;

pub trait B where
    Self: not_object_safe::A,
{
    fn f2(&self);
}

struct S(Box<dyn B>);
//~^ ERROR the trait `B` cannot be made into an object

fn main() {}
