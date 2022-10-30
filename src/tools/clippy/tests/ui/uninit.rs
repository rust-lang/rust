#![feature(stmt_expr_attributes)]
#![allow(clippy::let_unit_value, invalid_value)]

use std::mem::{self, MaybeUninit};

fn main() {
    let _: usize = unsafe { MaybeUninit::uninit().assume_init() };

    // edge case: For now we lint on empty arrays
    let _: [u8; 0] = unsafe { MaybeUninit::uninit().assume_init() };

    // edge case: For now we accept unit tuples
    let _: () = unsafe { MaybeUninit::uninit().assume_init() };

    // This is OK, because `MaybeUninit` allows uninitialized data.
    let _: MaybeUninit<usize> = unsafe { MaybeUninit::uninit().assume_init() };

    // This is OK, because all constitutent types are uninit-compatible.
    let _: (MaybeUninit<usize>, MaybeUninit<bool>) = unsafe { MaybeUninit::uninit().assume_init() };

    // This is OK, because all constitutent types are uninit-compatible.
    let _: (MaybeUninit<usize>, [MaybeUninit<bool>; 2]) = unsafe { MaybeUninit::uninit().assume_init() };

    // Was a false negative.
    let _: usize = unsafe { mem::MaybeUninit::uninit().assume_init() };
}
