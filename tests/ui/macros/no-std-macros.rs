//@ compile-flags: --crate-type=lib
//@ check-pass
// issue #55482
#![no_std]

macro_rules! foo {
    ($e:expr) => {
        $crate::core::assert!($e);
        $crate::core::assert_eq!($e, true);
    };
}

pub fn foo() { foo!(true); }
