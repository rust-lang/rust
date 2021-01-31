// check-pass

#![feature(trusted_len)]

use std::{
    array::IntoIter,
    fmt::Debug,
    iter::{ExactSizeIterator, FusedIterator, TrustedLen},
};

pub fn yes_iterator() -> impl Iterator<Item = i32> {
    IntoIter::new([0i32; 33])
}

pub fn yes_double_ended_iterator() -> impl DoubleEndedIterator {
    IntoIter::new([0i32; 33])
}

pub fn yes_exact_size_iterator() -> impl ExactSizeIterator {
    IntoIter::new([0i32; 33])
}

pub fn yes_fused_iterator() -> impl FusedIterator {
    IntoIter::new([0i32; 33])
}

pub fn yes_trusted_len() -> impl TrustedLen {
    IntoIter::new([0i32; 33])
}

pub fn yes_clone() -> impl Clone {
    IntoIter::new([0i32; 33])
}

pub fn yes_debug() -> impl Debug {
    IntoIter::new([0i32; 33])
}


fn main() {}
