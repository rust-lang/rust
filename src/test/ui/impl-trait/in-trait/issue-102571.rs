#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

use std::fmt::Display;
use std::ops::Deref;

trait Foo {
    fn bar(self) -> impl Deref<Target = impl Display + ?Sized>;
}

struct A;

impl Foo for A {
    fn bar(self) -> &'static str {
        "Hello, world"
    }
}

fn foo<T: Foo>(t: T) {
    let () = t.bar();
    //~^ ERROR mismatched types
}

fn main() {}
