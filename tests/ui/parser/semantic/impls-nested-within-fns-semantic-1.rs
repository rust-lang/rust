// Regression test for issue #119924.
// check-pass
#![feature(const_trait_impl, effects)]

pub struct Type;

#[const_trait]
trait Trait {
    fn required();
}

impl const Trait for () {
    fn required() {
        impl Type {
            // This visibility qualifier used to get rejected.
            pub fn perform() {}
        }
    }
}

fn main() {}
