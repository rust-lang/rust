// revisions: full min
#![allow(incomplete_features)]
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

trait HasSize {
    const SIZE: usize;
}

impl<const X: usize> HasSize for ArrayHolder<X> {
    const SIZE: usize = X;
}

struct ArrayHolder<const X: usize>([u32; X]);

impl<const X: usize> ArrayHolder<X> {
    pub const fn new() -> Self {
        ArrayHolder([0; Self::SIZE])
        //~^ ERROR constant expression depends on a generic parameter
        //[min]~| ERROR mismatched types
    }
}

fn main() {
    let mut array = ArrayHolder::new();
}
