// Regression test for #69615.

#![feature(const_trait_impl, const_fn)]
#![allow(incomplete_features)]

pub trait MyTrait {
    fn method(&self);
}

impl const MyTrait for () {
    fn method(&self) {
        loop {} //~ ERROR `loop` is not allowed in a `const fn`
    }
}

fn main() {}
