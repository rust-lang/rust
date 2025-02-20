#![feature(stmt_expr_attributes)]
#![allow(clippy::let_unit_value, invalid_value)]

use std::mem::MaybeUninit;

union MyOwnMaybeUninit {
    value: u8,
    uninit: (),
}

fn main() {
    let _: usize = unsafe { MaybeUninit::uninit().assume_init() };
    //~^ uninit_assumed_init

    // This is OK, because ZSTs do not contain data.
    let _: () = unsafe { MaybeUninit::uninit().assume_init() };

    // This is OK, because `MaybeUninit` allows uninitialized data.
    let _: MaybeUninit<usize> = unsafe { MaybeUninit::uninit().assume_init() };

    // This is OK, because all constituent types are uninit-compatible.
    let _: (MaybeUninit<usize>, MaybeUninit<bool>) = unsafe { MaybeUninit::uninit().assume_init() };

    // This is OK, because all constituent types are uninit-compatible.
    let _: (MaybeUninit<usize>, [MaybeUninit<bool>; 2]) = unsafe { MaybeUninit::uninit().assume_init() };

    // This is OK, because our own MaybeUninit is just as fine as the one from core.
    let _: MyOwnMaybeUninit = unsafe { MaybeUninit::uninit().assume_init() };

    // This is OK, because empty arrays don't contain data.
    let _: [u8; 0] = unsafe { MaybeUninit::uninit().assume_init() };

    // Was a false negative.
    let _: usize = unsafe { MaybeUninit::uninit().assume_init() };
    //~^ uninit_assumed_init

    polymorphic::<()>();
    polymorphic_maybe_uninit_array::<10>();
    polymorphic_maybe_uninit::<u8>();

    fn polymorphic<T>() {
        // We are conservative around polymorphic types.
        let _: T = unsafe { MaybeUninit::uninit().assume_init() };
        //~^ uninit_assumed_init
    }

    fn polymorphic_maybe_uninit_array<const N: usize>() {
        // While the type is polymorphic, MaybeUninit<u8> is not.
        let _: [MaybeUninit<u8>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    }

    fn polymorphic_maybe_uninit<T>() {
        // The entire type is polymorphic, but it's wrapped in a MaybeUninit.
        let _: MaybeUninit<T> = unsafe { MaybeUninit::uninit().assume_init() };
    }
}
