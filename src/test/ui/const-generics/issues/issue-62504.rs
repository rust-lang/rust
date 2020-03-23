// Regression test for #62504

#![feature(const_generics)]
#![allow(incomplete_features)]

trait HasSize {
    const SIZE: usize;
}

impl<const X: usize> HasSize for ArrayHolder<{ X }> {
    const SIZE: usize = X;
}

struct ArrayHolder<const X: usize>([u32; X]);

impl<const X: usize> ArrayHolder<{ X }> {
    pub const fn new() -> Self {
        ArrayHolder([0; Self::SIZE])
        //~^ ERROR: array lengths can't depend on generic parameters
    }
}

fn main() {
    let mut array = ArrayHolder::new();
}
