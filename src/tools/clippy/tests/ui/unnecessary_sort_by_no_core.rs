//@check-pass
#![feature(no_core)]
#![no_std]
#![no_core]
extern crate alloc;
extern crate core as mycore;
use alloc::vec;
use alloc::vec::Vec;
use mycore::cmp::Ord as _;

fn issue_11524() -> Vec<i32> {
    let mut vec = vec![1, 2, 3];

    // We could lint and suggest `vec.sort_by_key(|a| a + 1);`, but we don't bother to -- see the
    // comment in the lint at line 194
    vec.sort_by(|a, b| (a + 1).cmp(&(b + 1)));
    vec
}

fn issue_11524_2() -> Vec<i32> {
    let mut vec = vec![1, 2, 3];

    // Should not lint, as even `vec.sort_by_key(|b| core::cmp::Reverse(b + 1));` would not compile
    vec.sort_by(|a, b| (b + 1).cmp(&(a + 1)));
    vec
}
