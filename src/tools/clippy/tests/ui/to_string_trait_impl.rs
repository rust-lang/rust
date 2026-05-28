#![warn(clippy::to_string_trait_impl)]
#![feature(min_specialization)]

use std::fmt::{self, Display};

struct Point {
    x: usize,
    y: usize,
}

impl ToString for Point {
    //~^ to_string_trait_impl
    fn to_string(&self) -> String {
        format!("({}, {})", self.x, self.y)
    }
}

struct Foo;

impl Display for Foo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Foo")
    }
}

struct Bar;

impl Bar {
    #[allow(clippy::inherent_to_string)]
    fn to_string(&self) -> String {
        String::from("Bar")
    }
}
