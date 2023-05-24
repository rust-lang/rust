// check-pass

#![feature(return_position_impl_trait_in_trait)]
#![feature(refine)]
#![allow(incomplete_features)]

use std::fmt::Display;
use std::ops::Deref;

trait Foo {
    fn bar(self) -> impl Deref<Target = impl Display + ?Sized>;
}

struct A;

impl Foo for A {
    #[refine]
    fn bar(self) -> &'static str {
        "Hello, world"
    }
}

struct B;

impl Foo for B {
    #[refine]
    fn bar(self) -> Box<i32> {
        Box::new(42)
    }
}

fn main() {
    println!("Message for you: {:?}", &*A.bar());
    println!("Another for you: {:?}", &*B.bar());
}
