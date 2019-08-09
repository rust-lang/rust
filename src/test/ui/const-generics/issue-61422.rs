// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

use std::mem;

fn foo<const SIZE: usize>() {
    let arr: [u8; SIZE] = unsafe {
        #[allow(deprecated)]
        let mut array: [u8; SIZE] = mem::uninitialized();
        array
    };
}

fn main() {}
