#![allow(incomplete_features)]
#![feature(generic_associated_types)]

// Checking the interaction with this other feature
#![feature(associated_type_defaults)]

use std::fmt::{Display, Debug};

trait Foo {
    type Assoc where Self: Sized;
    type Assoc2<T> where T: Display;
    //~^ ERROR type-generic associated types are not yet implemented
    type Assoc3<T>;
    //~^ ERROR type-generic associated types are not yet implemented
    type WithDefault<'a, T: Debug + 'a> = dyn Iterator<Item=T>;
    //~^ ERROR type-generic associated types are not yet implemented
    type NoGenerics;
}

struct Bar;

impl Foo for Bar {
    type Assoc = usize;
    type Assoc2<T> = Vec<T>;
    type Assoc3<T> where T: Iterator = Vec<T>;
    type WithDefault<'a, T: Debug + 'a> = &'a dyn Iterator<Item=T>;
    type NoGenerics = ::std::cell::Cell<i32>;
}

fn main() {}
