//@aux-build:proc_macros.rs:proc-macro
#![allow(clippy::no_effect, clippy::useless_vec, unused)]
#![warn(clippy::single_range_in_vec_init)]
#![feature(generic_arg_infer)]

#[macro_use]
extern crate proc_macros;

macro_rules! a {
    () => {
        vec![0..200];
    };
}

fn awa<T: PartialOrd>(start: T, end: T) {
    [start..end];
}

fn awa_vec<T: PartialOrd>(start: T, end: T) {
    vec![start..end];
}

fn main() {
    // Lint
    [0..200];
    vec![0..200];
    [0u8..200];
    [0usize..200];
    [0..200usize];
    vec![0u8..200];
    vec![0usize..200];
    vec![0..200usize];
    // Only suggest collect
    [0..200isize];
    vec![0..200isize];
    // Do not lint
    [0..200, 0..100];
    vec![0..200, 0..100];
    [0.0..200.0];
    vec![0.0..200.0];
    // `Copy` is not implemented for `Range`, so this doesn't matter
    // FIXME: [0..200; 2];
    // FIXME: [vec!0..200; 2];

    // Unfortunately skips any macros
    a!();

    // Skip external macros and procedural macros
    external! {
        [0..200];
        vec![0..200];
    }
    with_span! {
        span
        [0..200];
        vec![0..200];
    }
}
