use std::cell::UnsafeCell;
use std::marker::PhantomData;

// This checks that `PhantomData` does not entirely mask interior mutability.

trait Trait {
    const C: (u32, PhantomData<UnsafeCell<u32>>);
}

fn bar<T: Trait>() {
    let x: &'static (u32, PhantomData<UnsafeCell<u32>>) = &T::C;
    //~^ ERROR: dropped while borrowed
}

fn main() {}
