//@ check-pass

use std::fmt::Display;

pub trait Foo {
    fn bar(&self) -> impl Display;
}

impl Foo for i32 {
    #[expect(refining_impl_trait)]
    fn bar(&self) -> i32 {
        *self
    }
}

impl Foo for &'static str {
    #[expect(refining_impl_trait)]
    fn bar(&self) -> &'static str {
        *self
    }
}

pub struct Yay;

impl Foo for Yay {
    #[expect(refining_impl_trait)]
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
