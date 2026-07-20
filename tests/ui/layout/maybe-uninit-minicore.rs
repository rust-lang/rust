//@ add-minicore
//@ check-pass

#![feature(no_core)]
#![no_core]
#![no_std]
#![crate_type = "lib"]

extern crate minicore;

use minicore::{MaybeUninit, mem};

// Regression test for https://github.com/rust-lang/rust/issues/159427.
// `minicore` has `MaybeUninit` and `ManuallyDrop`, but no `MaybeDangling`.
const _: usize = mem::size_of::<MaybeUninit<u8>>();
const _: usize = mem::align_of::<MaybeUninit<u8>>();
