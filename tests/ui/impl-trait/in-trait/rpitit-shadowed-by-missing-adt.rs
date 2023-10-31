// issue: 113903

#![feature(return_position_impl_trait_in_trait)]

use std::ops::Deref;

pub trait Tr {
    fn w() -> impl Deref<Target = Missing<impl Sized>>;
    //~^ ERROR cannot find type `Missing` in this scope
}

impl Tr for () {
    fn w() -> &'static () {
        &()
    }
}

fn main() {}
