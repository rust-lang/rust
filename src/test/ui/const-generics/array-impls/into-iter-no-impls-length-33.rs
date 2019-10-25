#![feature(array_value_iter)]
#![feature(trusted_len)]

use std::{
    array::IntoIter,
    fmt::Debug,
    iter::{ExactSizeIterator, FusedIterator, TrustedLen},
};

pub fn no_iterator() -> impl Iterator<Item = i32> {
    //~^ ERROR arrays only have std trait implementations for lengths 0..=32
    IntoIter::new([0i32; 33])
    //~^ ERROR arrays only have std trait implementations for lengths 0..=32
}

pub fn no_double_ended_iterator() -> impl DoubleEndedIterator {
    //~^ ERROR arrays only have std trait implementations for lengths 0..=32
    IntoIter::new([0i32; 33])
    //~^ ERROR arrays only have std trait implementations for lengths 0..=32
}

pub fn no_exact_size_iterator() -> impl ExactSizeIterator {
    //~^ ERROR arrays only have std trait implementations for lengths 0..=32
    IntoIter::new([0i32; 33])
    //~^ ERROR arrays only have std trait implementations for lengths 0..=32
}

pub fn no_fused_iterator() -> impl FusedIterator {
    //~^ ERROR arrays only have std trait implementations for lengths 0..=32
    IntoIter::new([0i32; 33])
    //~^ ERROR arrays only have std trait implementations for lengths 0..=32
}

pub fn no_trusted_len() -> impl TrustedLen {
    //~^ ERROR arrays only have std trait implementations for lengths 0..=32
    IntoIter::new([0i32; 33])
    //~^ ERROR arrays only have std trait implementations for lengths 0..=32
}

pub fn no_clone() -> impl Clone {
    //~^ ERROR arrays only have std trait implementations for lengths 0..=32
    IntoIter::new([0i32; 33])
    //~^ ERROR arrays only have std trait implementations for lengths 0..=32
}

pub fn no_debug() -> impl Debug {
    //~^ ERROR arrays only have std trait implementations for lengths 0..=32
    IntoIter::new([0i32; 33])
    //~^ ERROR arrays only have std trait implementations for lengths 0..=32
}


fn main() {}
