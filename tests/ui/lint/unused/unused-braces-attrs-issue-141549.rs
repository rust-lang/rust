//@ check-pass
//@ run-rustfix

#![allow(dead_code)]
#![warn(unused_braces)]

use std::cmp::Ordering;

#[rustfmt::skip]
fn ptr_cmp<T: ?Sized>(p1: *const T, p2: *const T) -> Ordering {
    { #[expect(ambiguous_wide_pointer_comparisons)] p1.cmp(&p2) }
    //~^ WARN unnecessary braces around block return value
}

fn main() {}
