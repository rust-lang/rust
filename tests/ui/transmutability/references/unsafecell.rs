#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

use std::cell::UnsafeCell;

mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn is_maybe_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, { Assume::SAFETY }>
    {}
}

fn value_to_value() {
    // We accept value-to-value transmutations of `UnsafeCell`-containing types,
    // because owning a value implies exclusive access.
    assert::is_maybe_transmutable::<UnsafeCell<u8>, u8>();
    assert::is_maybe_transmutable::<u8, UnsafeCell<u8>>();
    assert::is_maybe_transmutable::<UnsafeCell<u8>, UnsafeCell<u8>>();
}

fn ref_to_ref() {
    // We forbid `UnsafeCell`-containing ref-to-ref transmutations, because the
    // two types may use different, incompatible synchronization strategies.
    assert::is_maybe_transmutable::<&'static u8, &'static UnsafeCell<u8>>(); //~ ERROR: cannot be safely transmuted

    assert::is_maybe_transmutable::<&'static UnsafeCell<u8>, &'static UnsafeCell<u8>>(); //~ ERROR: cannot be safely transmuted
}

fn mut_to_mut() {
    // `UnsafeCell` does't matter for `&mut T` to `&mut U`, since exclusive
    // borrows can't be used for shared access.
    assert::is_maybe_transmutable::<&'static mut u8, &'static mut UnsafeCell<u8>>();
    assert::is_maybe_transmutable::<&'static mut UnsafeCell<u8>, &'static mut u8>();
    assert::is_maybe_transmutable::<&'static mut UnsafeCell<u8>, &'static mut UnsafeCell<u8>>();
}

fn mut_to_ref() {
    // `&mut UnsafeCell` is irrelevant in the source.
    assert::is_maybe_transmutable::<&'static mut UnsafeCell<bool>, &'static u8>();
    // `&UnsafeCell` in forbidden in the destination, since the destination can be used to
    // invalidate a shadowed source reference.
    assert::is_maybe_transmutable::<&'static mut bool, &'static UnsafeCell<u8>>(); //~ ERROR: cannot be safely transmuted
    assert::is_maybe_transmutable::<&'static mut UnsafeCell<bool>, &'static UnsafeCell<u8>>(); //~ ERROR: cannot be safely transmuted
}
