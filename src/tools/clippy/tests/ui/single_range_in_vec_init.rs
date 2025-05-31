//@aux-build:proc_macros.rs
//@no-rustfix: overlapping suggestions
#![allow(clippy::no_effect, clippy::unnecessary_operation, clippy::useless_vec, unused)]
#![warn(clippy::single_range_in_vec_init)]

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
    //~^ single_range_in_vec_init
    vec![0..200];
    //~^ single_range_in_vec_init
    [0u8..200];
    //~^ single_range_in_vec_init
    [0usize..200];
    //~^ single_range_in_vec_init
    [0..200usize];
    //~^ single_range_in_vec_init
    vec![0u8..200];
    //~^ single_range_in_vec_init
    vec![0usize..200];
    //~^ single_range_in_vec_init
    vec![0..200usize];
    //~^ single_range_in_vec_init
    // Only suggest collect
    [0..200isize];
    //~^ single_range_in_vec_init
    vec![0..200isize];
    //~^ single_range_in_vec_init
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
