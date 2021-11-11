#![warn(clippy::if_chain_style)]
#![allow(clippy::no_effect, clippy::nonminimal_bool, clippy::missing_clippy_version_attribute)]

extern crate if_chain;

use if_chain::if_chain;

fn main() {
    if true {
        let x = "";
        // `if_chain!` inside `if`
        if_chain! {
            if true;
            if true;
            then {}
        }
    }
    if_chain! {
        if true
            // multi-line AND'ed conditions
            && false;
        if let Some(1) = Some(1);
        // `let` before `then`
        let x = "";
        then {
            ();
        }
    }
    if_chain! {
        // single `if` condition
        if true;
        then {
            let x = "";
            // nested if
            if true {}
        }
    }
    if_chain! {
        // starts with `let ..`
        let x = "";
        if let Some(1) = Some(1);
        then {
            let x = "";
            let x = "";
            // nested if_chain!
            if_chain! {
                if true;
                if true;
                then {}
            }
        }
    }
}

fn negative() {
    if true {
        ();
        if_chain! {
            if true;
            if true;
            then { (); }
        }
    }
    if_chain! {
        if true;
        let x = "";
        if true;
        then { (); }
    }
    if_chain! {
        if true;
        if true;
        then {
            if true { 1 } else { 2 }
        } else {
            3
        }
    };
    if true {
        if_chain! {
            if true;
            if true;
            then {}
        }
    } else if false {
        if_chain! {
            if true;
            if false;
            then {}
        }
    }
}
