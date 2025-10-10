//@ proc-macro: match-expander.rs
//@ ignore-backends: gcc
// Ensure that we don't point at macro invocation when providing inference contexts.

#[macro_use]
extern crate match_expander;

fn main() {
    match_expander::matcher!();
    //~^ ERROR: mismatched types
    //~| NOTE: expected `S`, found `bool`
    //~| NOTE: in this expansion of match_expander::matcher!
}
