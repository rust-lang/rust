//@aux-build:proc_macro_suspicious_else_formatting.rs

#![warn(clippy::suspicious_else_formatting, clippy::possible_missing_else)]
#![allow(
    clippy::if_same_then_else,
    clippy::let_unit_value,
    clippy::needless_if,
    clippy::needless_else
)]

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
    //~^ possible_missing_else
    }

    if foo() {
    } if foo() {
    //~^ possible_missing_else
    }

    let _ = { // if as the last expression
        let _ = 0;

        if foo() {
        } if foo() {
        //~^ possible_missing_else
        }
        else {
        }
    };

    let _ = { // if in the middle of a block
        if foo() {
        } if foo() {
        //~^ possible_missing_else
        }
        else {
        }

        let _ = 0;
    };

    if foo() {
    } else
    {
    }
    //~^^^ suspicious_else_formatting

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
    //~^^^ suspicious_else_formatting

    if foo() {
    }
    else
    if foo() { // the span of the above error should continue here
    }
    //~^^^^ suspicious_else_formatting

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
    //~^^^^^ suspicious_else_formatting

    if foo() {
    }
    else

    {

    }
    //~^^^^^^ suspicious_else_formatting

    // #3864 - Allman style braces
    if foo()
    {
    }
    else
    {
    }

    //#10273 This is fine. Don't warn
    if foo() {
    } else
    /* whelp */
    {
    }

    // #12497 Don't trigger lint as rustfmt wants it
    if true {
        println!("true");
    }
    /*else if false {
}*/
    else {
        println!("false");
    }

    if true {
        println!("true");
    } // else if false {}
    else {
        println!("false");
    }

    if true {
        println!("true");
    } /* if true {
        println!("true");
}
    */
    else {
        println!("false");
    }

}

// #7650 - Don't lint. Proc-macro using bad spans for `if` expressions.
#[derive(DeriveBadSpan)]
struct _Foo(u32, u32);
