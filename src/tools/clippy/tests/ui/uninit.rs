#![feature(stmt_expr_attributes)]
#![allow(clippy::let_unit_value, invalid_value)]

use std::mem::{self, MaybeUninit};

union MyOwnMaybeUninit {
    value: u8,
    uninit: (),
}

fn main() {
    let _: usize = unsafe { MaybeUninit::uninit().assume_init() };

    // This is OK, because ZSTs do not contain data.
    let _: () = unsafe { MaybeUninit::uninit().assume_init() };

    // This is OK, because `MaybeUninit` allows uninitialized data.
    let _: MaybeUninit<usize> = unsafe { MaybeUninit::uninit().assume_init() };

    // This is OK, because all constitutent types are uninit-compatible.
    let _: (MaybeUninit<usize>, MaybeUninit<bool>) = unsafe { MaybeUninit::uninit().assume_init() };

    // This is OK, because all constitutent types are uninit-compatible.
    let _: (MaybeUninit<usize>, [MaybeUninit<bool>; 2]) = unsafe { MaybeUninit::uninit().assume_init() };

    // This is OK, because our own MaybeUninit is just as fine as the one from core.
    let _: MyOwnMaybeUninit = unsafe { MaybeUninit::uninit().assume_init() };

    // This is OK, because empty arrays don't contain data.
    let _: [u8; 0] = unsafe { MaybeUninit::uninit().assume_init() };

    // Was a false negative.
    let _: usize = unsafe { mem::MaybeUninit::uninit().assume_init() };

    polymorphic::<()>();

    fn polymorphic<T>() {
        // We are conservative around polymorphic types.
        let _: T = unsafe { mem::MaybeUninit::uninit().assume_init() };
    }
}
