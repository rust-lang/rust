//@aux-build:proc_macros.rs
#![allow(clippy::useless_vec, clippy::iter_out_of_bounds, unused)]
#![warn(clippy::iter_skip_zero)]

#[macro_use]
extern crate proc_macros;

use std::iter::once;

fn main() {
    let _ = [1, 2, 3].iter().skip(0);
    let _ = vec![1, 2, 3].iter().skip(0);
    let _ = once([1, 2, 3]).skip(0);
    let _ = vec![1, 2, 3].iter().chain([1, 2, 3].iter().skip(0)).skip(0);
    // Don't lint
    let _ = [1, 2, 3].iter().skip(1);
    let _ = vec![1, 2, 3].iter().skip(1);
    external! {
        let _ = [1, 2, 3].iter().skip(0);
    }
    with_span! {
        let _ = [1, 2, 3].iter().skip(0);
    }
}
