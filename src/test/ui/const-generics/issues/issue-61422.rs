// check-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))] //[full]~WARN the feature `const_generics` is incomplete

use std::mem;

// Neither of the uninits below are currently accepted as not UB, however,
// this code does not run and is merely checking that we do not ICE on this pattern,
// so this is fine.

fn foo<const SIZE: usize>() {
    let arr: [u8; SIZE] = unsafe {
        #[allow(deprecated)]
        let array: [u8; SIZE] = mem::uninitialized();
        array
    };
}

fn bar<const SIZE: usize>() {
    let arr: [u8; SIZE] = unsafe {
        let array: [u8; SIZE] = mem::MaybeUninit::uninit().assume_init();
        array
    };
}


fn main() {}
