// aux-build:proc_macro_suspicious_else_formatting.rs

#![warn(clippy::suspicious_else_formatting)]
#![allow(clippy::if_same_then_else)]

extern crate proc_macro_suspicious_else_formatting;
use proc_macro_suspicious_else_formatting::DeriveBadSpan;

fn foo() -> bool {
    true
}

#[rustfmt::skip]
fn main() {
    // weird `else` formatting:
    if foo() {
    } {
    }

    if foo() {
    } if foo() {
    }

    let _ = { // if as the last expression
        let _ = 0;

        if foo() {
        } if foo() {
        }
        else {
        }
    };

    let _ = { // if in the middle of a block
        if foo() {
        } if foo() {
        }
        else {
        }

        let _ = 0;
    };

    if foo() {
    } else
    {
    }

    // This is fine, though weird. Allman style braces on the else.
    if foo() {
    }
    else
    {
    }

    if foo() {
    } else
    if foo() { // the span of the above error should continue here
    }

    if foo() {
    }
    else
    if foo() { // the span of the above error should continue here
    }

    // those are ok:
    if foo() {
    }
    {
    }

    if foo() {
    } else {
    }

    if foo() {
    }
    else {
    }

    if foo() {
    }
    if foo() {
    }

    // Almost Allman style braces. Lint these.
    if foo() {
    }

    else
    {

    }

    if foo() {
    }
    else

    {

    }

    // #3864 - Allman style braces
    if foo()
    {
    }
    else
    {
    }
}

// #7650 - Don't lint. Proc-macro using bad spans for `if` expressions.
#[derive(DeriveBadSpan)]
struct _Foo(u32, u32);
