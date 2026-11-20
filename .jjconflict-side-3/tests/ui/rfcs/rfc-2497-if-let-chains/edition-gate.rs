// Here we test Rust 2024 edition gating for ´let_chains`.
// See `disallowed-positions.rs` for the grammar
// defining the language for gated allowed positions.

#![allow(irrefutable_let_patterns)]

use std::ops::Range;

fn _if() {
    if let 0 = 1 {} // Stable!

    if true && let 0 = 1 {}
    //~^ ERROR let chains are only allowed in Rust 2024 or later

    if let 0 = 1 && true {}
    //~^ ERROR let chains are only allowed in Rust 2024 or later

    if let Range { start: _, end: _ } = (true..true) && false {}
    //~^ ERROR let chains are only allowed in Rust 2024 or later

    if let 1 = 1 && let true = { true } && false {
    //~^ ERROR let chains are only allowed in Rust 2024 or later
    //~| ERROR let chains are only allowed in Rust 2024 or later
    }
}

fn _while() {
    while let 0 = 1 {} // Stable!

    while true && let 0 = 1 {}
    //~^ ERROR let chains are only allowed in Rust 2024 or later

    while let 0 = 1 && true {}
    //~^ ERROR let chains are only allowed in Rust 2024 or later

    while let Range { start: _, end: _ } = (true..true) && false {}
    //~^ ERROR let chains are only allowed in Rust 2024 or later
}

fn _macros() {
    macro_rules! noop_expr { ($e:expr) => {}; }

    noop_expr!((let 0 = 1));
    //~^ ERROR expected expression, found `let` statement

    macro_rules! use_expr {
        ($e:expr) => {
            if $e {}
            while $e {}
        }
    }
    #[cfg(false)] (let 0 = 1);
    //~^ ERROR expected expression, found `let` statement
    use_expr!(let 0 = 1);
    //~^ ERROR no rules expected keyword `let`
}

fn main() {}
