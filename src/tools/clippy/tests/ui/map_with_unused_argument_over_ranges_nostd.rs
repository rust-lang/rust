#![warn(clippy::map_with_unused_argument_over_ranges)]
#![no_std]
extern crate alloc;
use alloc::vec::Vec;

fn nostd(v: &mut [i32]) {
    let _: Vec<_> = (0..10).map(|_| 3 + 1).collect();
    //~^ map_with_unused_argument_over_ranges
}
