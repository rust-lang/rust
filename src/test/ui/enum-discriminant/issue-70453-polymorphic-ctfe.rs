// run-pass
#![feature(arbitrary_enum_discriminant, core_intrinsics)]

extern crate core;
use core::intrinsics::discriminant_value;

#[repr(usize)]
enum MyWeirdOption<T> {
    None = 0,
    Some(T) = core::mem::size_of::<*mut T>(),
    //~^ WARN cannot use constants which depend on generic parameters in types
    //~| WARN this was previously accepted by the compiler but is being phased out
}

fn main() {
    assert_eq!(discriminant_value(&MyWeirdOption::<()>::None), 0);
    assert_eq!(discriminant_value(&MyWeirdOption::Some(())), core::mem::size_of::<usize>());
}
