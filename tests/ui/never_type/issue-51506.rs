#![feature(never_type, specialization)]
#![allow(incomplete_features)]

use std::iter::{self, Empty};

trait Trait {
    type Out: Iterator<Item = u32>;

    fn f(&self) -> Option<Self::Out>;
}

impl<T> Trait for T {
    default type Out = !; //~ ERROR: `!` is not an iterator

    default fn f(&self) -> Option<Self::Out> {
        None
    }
}

struct X;

impl Trait for X {
    type Out = Empty<u32>;

    fn f(&self) -> Option<Self::Out> {
        Some(iter::empty())
    }
}

fn f<T: Trait>(a: T) {
    if let Some(iter) = a.f() {
        println!("Some");
        for x in iter {
            println!("x = {}", x);
        }
    }
}

pub fn main() {
    f(10);
}
