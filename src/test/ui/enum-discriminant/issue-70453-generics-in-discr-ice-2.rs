#![feature(arbitrary_enum_discriminant, core_intrinsics)]

extern crate core;
use core::intrinsics::discriminant_value;

#[repr(usize)]
enum MyWeirdOption<T> {
    None = 0,
    Some(T) = std::mem::size_of::<T>(),
    //~^ ERROR generic parameters may not be used in const operations
}

fn main() {
    assert_eq!(discriminant_value(&MyWeirdOption::<u8>::None), 0);
    assert_eq!(discriminant_value(&MyWeirdOption::Some(0u8)), 1);
}
