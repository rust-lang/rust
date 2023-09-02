// check-pass

#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

use std::fmt::Display;

trait Foo {
    fn bar(&self) -> impl Display;
}

impl Foo for i32 {
    #[allow(refining_impl_trait)]
    fn bar(&self) -> i32 {
        *self
    }
}

impl Foo for &'static str {
    #[allow(refining_impl_trait)]
    fn bar(&self) -> &'static str {
        *self
    }
}

struct Yay;

impl Foo for Yay {
    #[allow(refining_impl_trait)]
    fn bar(&self) -> String {
        String::from(":^)")
    }
}

fn foo_generically<T: Foo>(t: T) {
    println!("{}", t.bar());
}

fn main() {
    println!("{}", "Hello, world.".bar());
    println!("The answer is {}!", 42.bar());
    foo_generically(Yay);
}
