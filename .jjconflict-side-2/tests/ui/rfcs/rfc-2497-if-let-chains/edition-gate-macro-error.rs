//@ revisions: edition2021 edition2024
//@ compile-flags: -Z lint-mir -Z validate-mir
//@ [edition2021] edition: 2021
//@ [edition2024] edition: 2024
//@ aux-build:macro-in-2021.rs
//@ aux-build:macro-in-2024.rs

use std::unreachable as never;

// Compiletest doesn't specify the needed --extern flags to make `extern crate` unneccessary
extern crate macro_in_2021;
extern crate macro_in_2024;

fn main() {
    // Gated on both 2021 and 2024 if the `if` comes from a 2021 macro
    // Gated only on 2021 if the `if` comes from a 2024 macro
    // No gating if both the `if` and the chain are from a 2024 macro

    macro_in_2021::make_if!((let Some(0) = None && let Some(0) = None) { never!() } { never!() });
    //~^ ERROR let chains are only allowed in Rust 2024 or later
    //~| ERROR let chains are only allowed in Rust 2024 or later
    macro_in_2021::make_if!(let (Some(0)) let (Some(0)) { never!() } { never!() });
    //~^ ERROR let chains are only allowed in Rust 2024 or later
    //~| ERROR let chains are only allowed in Rust 2024 or later

    macro_in_2024::make_if!((let Some(0) = None && let Some(0) = None) { never!() } { never!() });
    //[edition2021]~^ ERROR let chains are only allowed in Rust 2024 or later
    //[edition2021]~| ERROR let chains are only allowed in Rust 2024 or later
    macro_in_2024::make_if!(let (Some(0)) let (Some(0)) { never!() } { never!() });
}
