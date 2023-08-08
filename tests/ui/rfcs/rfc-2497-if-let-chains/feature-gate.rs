// gate-test-let_chains

// Here we test feature gating for ´let_chains`.
// See `disallowed-positions.rs` for the grammar
// defining the language for gated allowed positions.

#![allow(irrefutable_let_patterns)]

use std::ops::Range;

fn _if() {
    if let 0 = 1 {} // Stable!

    if true && let 0 = 1 {}
    //~^ ERROR `let` expressions in this position are unstable [E0658]

    if let 0 = 1 && true {}
    //~^ ERROR `let` expressions in this position are unstable [E0658]

    if let Range { start: _, end: _ } = (true..true) && false {}
    //~^ ERROR `let` expressions in this position are unstable [E0658]

    if let 1 = 1 && let true = { true } && false {
    //~^ ERROR `let` expressions in this position are unstable [E0658]
    //~| ERROR `let` expressions in this position are unstable [E0658]
    }
}

fn _while() {
    while let 0 = 1 {} // Stable!

    while true && let 0 = 1 {}
    //~^ ERROR `let` expressions in this position are unstable [E0658]

    while let 0 = 1 && true {}
    //~^ ERROR `let` expressions in this position are unstable [E0658]

    while let Range { start: _, end: _ } = (true..true) && false {}
    //~^ ERROR `let` expressions in this position are unstable [E0658]
}

fn _macros() {
    macro_rules! noop_expr { ($e:expr) => {}; }

    noop_expr!((let 0 = 1));
    //~^ ERROR `let` expressions in this position are unstable [E0658]
    //~| ERROR expected expression, found `let` statement

    macro_rules! use_expr {
        ($e:expr) => {
            if $e {}
            while $e {}
        }
    }
    #[cfg(FALSE)] (let 0 = 1);
    //~^ ERROR `let` expressions in this position are unstable [E0658]
    //~| ERROR expected expression, found `let` statement
    use_expr!(let 0 = 1);
    //~^ ERROR no rules expected the token `let`
}

fn main() {}
