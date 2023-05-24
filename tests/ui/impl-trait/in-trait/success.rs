// check-pass

#![feature(return_position_impl_trait_in_trait, refine)]
#![allow(incomplete_features)]

use std::fmt::Display;

trait Foo {
    fn bar(&self) -> impl Display;
}

impl Foo for i32 {
    #[refine]
    fn bar(&self) -> i32 {
        *self
    }
}

impl Foo for &'static str {
    #[refine]
    fn bar(&self) -> &'static str {
        *self
    }
}

struct Yay;

impl Foo for Yay {
    #[refine]
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
