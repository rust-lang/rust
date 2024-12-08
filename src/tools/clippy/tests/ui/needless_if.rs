//@aux-build:proc_macros.rs
#![feature(let_chains)]
#![allow(
    clippy::blocks_in_conditions,
    clippy::if_same_then_else,
    clippy::ifs_same_cond,
    clippy::let_unit_value,
    clippy::needless_else,
    clippy::no_effect,
    clippy::nonminimal_bool,
    clippy::short_circuit_statement,
    clippy::unnecessary_operation,
    clippy::redundant_pattern_matching,
    unused
)]
#![warn(clippy::needless_if)]

extern crate proc_macros;
use proc_macros::{external, with_span};

fn maybe_side_effect() -> bool {
    true
}

fn main() {
    // Lint
    if (true) {}
    // Do not remove the condition
    if maybe_side_effect() {}
    // Do not lint
    if (true) {
    } else {
    }
    if {
        return;
    } {}
    // Do not lint if `else if` is present
    if (true) {
    } else if (true) {
    }
    // Do not lint `if let` or let chains
    if let true = true {}
    if let true = true
        && true
    {}
    if true
        && let true = true
    {}
    // Can lint nested `if let`s
    if {
        if let true = true
            && true
        {
            true
        } else {
            false
        }
    } && true
    {}
    external! { if (true) {} }
    with_span! {
        span
        if (true) {}
    }

    if true {
        // comment
    }

    if true {
        #[cfg(any())]
        foo;
    }

    macro_rules! empty_expansion {
        () => {};
    }

    if true {
        empty_expansion!();
    }

    macro_rules! empty_repetition {
        ($($t:tt)*) => {
            if true {
                $($t)*
            }
        }
    }

    empty_repetition!();

    // Must be placed into an expression context to not be interpreted as a block
    if { maybe_side_effect() } {}
    // Would be a block followed by `&&true` - a double reference to `true`
    if { maybe_side_effect() } && true {}

    // Don't leave trailing attributes
    #[allow(unused)]
    if true {}

    let () = if maybe_side_effect() {};
}
