#![allow(
    clippy::assertions_on_constants,
    clippy::equatable_if_let,
    clippy::needless_if,
    clippy::nonminimal_bool,
    clippy::eq_op,
    clippy::redundant_pattern_matching
)]

#[rustfmt::skip]
#[warn(clippy::collapsible_if)]
fn main() {
    let x = "hello";
    let y = "world";
    if x == "hello" {
        if y == "world" {
            println!("Hello world!");
        }
    }
    //~^^^^^ collapsible_if

    if x == "hello" || x == "world" {
        if y == "world" || y == "hello" {
            println!("Hello world!");
        }
    }
    //~^^^^^ collapsible_if

    if x == "hello" && x == "world" {
        if y == "world" || y == "hello" {
            println!("Hello world!");
        }
    }
    //~^^^^^ collapsible_if

    if x == "hello" || x == "world" {
        if y == "world" && y == "hello" {
            println!("Hello world!");
        }
    }
    //~^^^^^ collapsible_if

    if x == "hello" && x == "world" {
        if y == "world" && y == "hello" {
            println!("Hello world!");
        }
    }
    //~^^^^^ collapsible_if

    if 42 == 1337 {
        if 'a' != 'A' {
            println!("world!")
        }
    }
    //~^^^^^ collapsible_if

    // Works because any if with an else statement cannot be collapsed.
    if x == "hello" {
        if y == "world" {
            println!("Hello world!");
        }
    } else {
        println!("Not Hello world");
    }

    if x == "hello" {
        if y == "world" {
            println!("Hello world!");
        } else {
            println!("Hello something else");
        }
    }

    if x == "hello" {
        print!("Hello ");
        if y == "world" {
            println!("world!")
        }
    }

    if true {
    } else {
        assert!(true); // assert! is just an `if`
    }

    if x == "hello" {
        if y == "world" { // Collapsible
            println!("Hello world!");
        }
    }
    //~^^^^^ collapsible_if

    if x == "hello" {
        print!("Hello ");
    } else {
        // Not collapsible
        if let Some(42) = Some(42) {
            println!("world!")
        }
    }

    if x == "hello" {
        print!("Hello ");
    } else {
        // Not collapsible
        if y == "world" {
            println!("world!")
        }
    }

    // Test behavior wrt. `let_chains`.
    // None of the cases below should be collapsed.
    fn truth() -> bool { true }

    // Prefix:
    if let 0 = 1 {
        if truth() {}
    }

    // Suffix:
    if truth() {
        if let 0 = 1 {}
    }

    // Midfix:
    if truth() {
        if let 0 = 1 {
            if truth() {}
        }
    }

    // Fix #5962
    if matches!(true, true) {
        if matches!(true, true) {}
    }
    //~^^^ collapsible_if

    // Issue #9375
    if matches!(true, true) && truth() {
        if matches!(true, true) {}
    }
    //~^^^ collapsible_if

    if true {
        #[cfg(not(teehee))]
        if true {
            println!("Hello world!");
        }
    }

    if true {
        if true {
            println!("No comment, linted");
        }
    }
    //~^^^^^ collapsible_if

    if true {
        // Do not collapse because of this comment
        if true {
            println!("Hello world!");
        }
    }
}

#[rustfmt::skip]
fn layout_check() -> u32 {
    if true {
        if true {
        }
        // This is a comment, do not collapse code to it
    }; 3
    //~^^^^^ collapsible_if
}
