//@ run-pass

#![allow(dead_code)]

use std::{alloc::Layout, ffi::c_int, mem::MaybeUninit, ptr};

// A simple uninhabited type.
enum Void {}

#[repr(C)]
enum Univariant<T> {
    Variant(T),
}

#[repr(C, u8)]
enum UnivariantU8<T> {
    Variant(T),
}

#[repr(C)]
enum TwoVariants<T> {
    Variant1(T),
    Variant2(u8),
}

#[repr(C, u8)]
enum TwoVariantsU8<T> {
    Variant1(T),
    Variant2(u8),
}

#[repr(C, u8)]
enum DeadBranchHasOtherField<T> {
    Variant1(T, u64),
    Variant2(u8),
}

macro_rules! assert_layout_eq {
    ($a:ty, $b:ty) => {
        assert_eq!(Layout::new::<$a>(), Layout::new::<$b>());
    };
}

fn main() {
    // Compiler must not remove dead variants of `#[repr(C)]` ADTs.
    assert_layout_eq!(Univariant<Void>, c_int);
    // This should also hold for `#[repr(C, int)]` ADTs.
    assert_layout_eq!(UnivariantU8<Void>, u8);
    // And for ADTs with more than one variant.
    // These are twice the size: a tag plus the field in a second branch.
    assert_layout_eq!(TwoVariants<Void>, [c_int; 2]);
    assert_layout_eq!(TwoVariantsU8<Void>, [u8; 2]);
    // This one is 2 x u64: we reserve space for fields in a dead branch.
    assert_layout_eq!(DeadBranchHasOtherField<Void>, [u64; 2]);

    // Some other useful invariants. See this UCG thread for more context:
    // https://github.com/rust-lang/unsafe-code-guidelines/issues/500
    assert_layout_eq!(Univariant<Void>, Univariant<MaybeUninit<Void>>);
    assert_layout_eq!(UnivariantU8<Void>, UnivariantU8<MaybeUninit<Void>>);
    assert_layout_eq!(TwoVariants<Void>, TwoVariants<MaybeUninit<Void>>);
    assert_layout_eq!(TwoVariantsU8<Void>, TwoVariantsU8<MaybeUninit<Void>>);
    assert_layout_eq!(DeadBranchHasOtherField<Void>, DeadBranchHasOtherField<MaybeUninit<Void>>);

    // Check that discriminants are allocated properly.
    // SAFETY:
    // 1. We checked that layout requires proper alignment.
    // 2. Discriminant is guaranteed to be the first field of a `#[repr(C)]` enum.
    unsafe {
        assert_eq!(*ptr::from_ref(&TwoVariants::<Void>::Variant2(42)).cast::<c_int>(), 1);
        assert_eq!(*ptr::from_ref(&TwoVariantsU8::<Void>::Variant2(42)).cast::<u8>(), 1);
        assert_eq!(*ptr::from_ref(&DeadBranchHasOtherField::<Void>::Variant2(42)).cast::<u8>(), 1);
    }
}
