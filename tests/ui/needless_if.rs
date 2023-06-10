//@run-rustfix
//@aux-build:proc_macros.rs
#![allow(
    clippy::blocks_in_if_conditions,
    clippy::if_same_then_else,
    clippy::ifs_same_cond,
    clippy::needless_else,
    clippy::no_effect,
    unused
)]
#![warn(clippy::needless_if)]

extern crate proc_macros;
use proc_macros::external;
use proc_macros::with_span;

fn main() {
    // Lint
    if (true) {}
    // Do not lint
    if (true) {
    } else {
    }
    // Do not lint if `else if` is present
    if (true) {
    } else if (true) {
    }
    // Ensure clippy does not bork this up, other cases should be added
    if {
        return;
    } {}
    external! { if (true) {} }
    with_span! {
        span
        if (true) {}
    }
}
