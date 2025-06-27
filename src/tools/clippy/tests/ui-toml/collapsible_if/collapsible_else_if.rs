#![allow(clippy::eq_op, clippy::nonminimal_bool)]

#[rustfmt::skip]
#[warn(clippy::collapsible_if)]
fn main() {
    let (x, y) = ("hello", "world");

    if x == "hello" {
        todo!()
    } else {
        // Comment must be kept
        if y == "world" {
            println!("Hello world!");
        }
    }
    //~^^^^^^ collapsible_else_if

    if x == "hello" {
        todo!()
    } else { // Inner comment
        if y == "world" {
            println!("Hello world!");
        }
    }
    //~^^^^^ collapsible_else_if

    if x == "hello" {
        todo!()
    } else {
        /* Inner comment */
        if y == "world" {
            println!("Hello world!");
        }
    }
    //~^^^^^^ collapsible_else_if

    if x == "hello" { 
        todo!()
    } else { /* Inner comment */
        if y == "world" {
            println!("Hello world!");
        }
    }
    //~^^^^^ collapsible_else_if

    if x == "hello" {
        todo!()
    } /* This should not be removed */ else /* So does this */ {
        // Comment must be kept
        if y == "world" {
            println!("Hello world!");
        }
    }
    //~^^^^^^ collapsible_else_if
}
