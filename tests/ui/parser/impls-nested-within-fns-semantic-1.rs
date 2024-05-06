// Regression test for part of issue #119924.
//@ check-pass

#![feature(const_trait_impl, effects)]

#[const_trait]
trait Trait {
    fn required();
}

impl const Trait for () {
    fn required() {
        pub struct Type;

        impl Type {
            // This visibility qualifier used to get rejected.
            pub fn perform() {}
        }
    }
}

fn main() {}
