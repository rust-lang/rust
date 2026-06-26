#![feature(core_intrinsics)]

extern crate core;
use core::intrinsics::discriminant_value;

#[repr(usize)]
enum MyWeirdOption<T> {
    None = 0,
    Some(T) = core::mem::size_of::<*mut T>(),
    //~^ ERROR generic parameters may not be used
}

fn main() {
    assert_eq!(discriminant_value(&MyWeirdOption::<()>::None), 0);
    assert_eq!(discriminant_value(&MyWeirdOption::Some(())), core::mem::size_of::<usize>());
}
