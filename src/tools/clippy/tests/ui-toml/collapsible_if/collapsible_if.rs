#![allow(clippy::eq_op, clippy::nonminimal_bool)]

#[rustfmt::skip]
#[warn(clippy::collapsible_if)]
fn main() {
    let (x, y) = ("hello", "world");

    if x == "hello" {
        // Comment must be kept
        if y == "world" {
            println!("Hello world!");
        }
    }
    //~^^^^^^ collapsible_if

    // The following tests check for the fix of https://github.com/rust-lang/rust-clippy/issues/798
    if x == "hello" { // Inner comment
        if y == "world" {
            println!("Hello world!");
        }
    }
    //~^^^^^ collapsible_if

    if x == "hello" {
        /* Inner comment */
        if y == "world" {
            println!("Hello world!");
        }
    }
    //~^^^^^^ collapsible_if

    if x == "hello" { /* Inner comment */
        if y == "world" {
            println!("Hello world!");
        }
    }
    //~^^^^^ collapsible_if
}
