#![feature(generic_associated_types)] //~ WARN `generic_associated_types` is incomplete

// Checking the interaction with this other feature
#![feature(associated_type_defaults)]

use std::fmt::{Display, Debug};

trait Foo {
    type Assoc where Self: Sized;
    type Assoc2<T> where T: Display;
    type Assoc3<T>;
    type WithDefault<T> where T: Debug = dyn Iterator<Item=T>;
    type NoGenerics;
}

struct Bar;

impl Foo for Bar {
    type Assoc = usize;
    type Assoc2<T> = Vec<T>;
    type Assoc3<T> where T: Iterator = Vec<T>;
    type WithDefault<'a, T> = &'a dyn Iterator<T>;
    type NoGenerics = ::std::cell::Cell<i32>;
}

fn main() {}
