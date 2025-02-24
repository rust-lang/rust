//! Ensure that we don't try to collect monomorphizeable items inside free const
//! items (their def site to be clear) whose generics require monomorphization.
//!
//! Such items are to be collected at instantiation sites of free consts.

#![feature(generic_const_items)]
#![allow(incomplete_features)]

//@ build-pass (we require monomorphization)

const _IDENTITY<T>: fn(T) -> T = |x| x;

fn main() {}
