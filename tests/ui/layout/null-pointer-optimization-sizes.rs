//! null pointer optimization preserves type sizes.
//!
//! Verifies that Option<T> has the same size as T for non-null pointer types,
//! and for custom enums that have a niche.

//@ run-pass

// Needs for Nothing variat in Enum
#![allow(dead_code)]

use std::mem;

enum E<T> {
    Thing(isize, T),
    Nothing((), ((), ()), [i8; 0]),
}

struct S<T>(isize, T);

// These are macros so we get useful assert messages.

macro_rules! check_option {
    ($T:ty) => {
        assert_eq!(mem::size_of::<Option<$T>>(), mem::size_of::<$T>());
    };
}

macro_rules! check_fancy {
    ($T:ty) => {
        assert_eq!(mem::size_of::<E<$T>>(), mem::size_of::<S<$T>>());
    };
}

macro_rules! check_type {
    ($T:ty) => {{
        check_option!($T);
        check_fancy!($T);
    }};
}

pub fn main() {
    check_type!(&'static isize);
    check_type!(Box<isize>);
    check_type!(extern "C" fn());
}
