#![feature(lazy_normalization_consts)]
//~^ WARN the feature `lazy_normalization_consts` is incomplete
trait ArraySizeTrait {
    const SIZE: usize = 0;
}

impl<T: ?Sized> ArraySizeTrait for T {
    const SIZE: usize = 1;
}

struct SomeArray<T: ArraySizeTrait> {
    array: [u8; T::SIZE],
    //~^ ERROR constant expression depends on a generic parameter
    phantom: std::marker::PhantomData<T>,
}

fn main() {}
