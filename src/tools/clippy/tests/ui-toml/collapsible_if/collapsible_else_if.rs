#![allow(clippy::eq_op, clippy::nonminimal_bool)]
#![warn(clippy::collapsible_if)]

#[rustfmt::skip]
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

fn issue_13365() {
    // the comments don't stop us from linting, so the the `expect` *will* be fulfilled
    if true {
    } else {
        // some other text before
        #[expect(clippy::collapsible_else_if)]
        if false {}
    }

    if true {
    } else {
        #[expect(clippy::collapsible_else_if)]
        // some other text after
        if false {}
    }
}
