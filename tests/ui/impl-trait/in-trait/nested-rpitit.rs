//@ check-pass

use std::fmt::Display;
use std::ops::Deref;

pub trait Foo {
    fn bar(self) -> impl Deref<Target = impl Display + ?Sized>;
}

pub struct A;

impl Foo for A {
    #[expect(refining_impl_trait)]
    fn bar(self) -> &'static str {
        "Hello, world"
    }
}

pub struct B;

impl Foo for B {
    #[expect(refining_impl_trait)]
    fn bar(self) -> Box<i32> {
        Box::new(42)
    }
}

fn main() {
    println!("Message for you: {:?}", &*A.bar());
    println!("Another for you: {:?}", &*B.bar());
}
