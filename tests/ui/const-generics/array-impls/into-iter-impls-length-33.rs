//@ check-pass

#![feature(trusted_len)]

use std::{
    array::IntoIter,
    fmt::Debug,
    iter::{ExactSizeIterator, FusedIterator, TrustedLen},
};

pub fn yes_iterator() -> impl Iterator<Item = i32> {
    IntoIterator::into_iter([0i32; 33])
}

pub fn yes_double_ended_iterator() -> impl DoubleEndedIterator {
    IntoIterator::into_iter([0i32; 33])
}

pub fn yes_exact_size_iterator() -> impl ExactSizeIterator {
    IntoIterator::into_iter([0i32; 33])
}

pub fn yes_fused_iterator() -> impl FusedIterator {
    IntoIterator::into_iter([0i32; 33])
}

pub fn yes_trusted_len() -> impl TrustedLen {
    IntoIterator::into_iter([0i32; 33])
}

pub fn yes_clone() -> impl Clone {
    IntoIterator::into_iter([0i32; 33])
}

pub fn yes_debug() -> impl Debug {
    IntoIterator::into_iter([0i32; 33])
}


fn main() {}
