//@ revisions: full min
#![allow(incomplete_features)]
#![cfg_attr(full, feature(generic_const_exprs))]
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
        //~^ ERROR mismatched types
        //[full]~^^ ERROR unconstrained generic constant
        //[min]~^^^ ERROR constant expression depends on a generic parameter
    }
}

fn main() {
    let mut array = ArrayHolder::new();
    //~^ ERROR: type annotations needed
    //~| ERROR type annotations needed
}
