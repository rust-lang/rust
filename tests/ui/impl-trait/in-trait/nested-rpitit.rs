// check-pass
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

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

struct B;

impl Foo for B {
    fn bar(self) -> Box<i32> {
        Box::new(42)
    }
}

fn main() {
    println!("Message for you: {:?}", &*A.bar());
    println!("Another for you: {:?}", &*B.bar());
}
