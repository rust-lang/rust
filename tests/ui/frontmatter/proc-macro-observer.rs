//@ check-pass
//@ proc-macro: makro.rs
//@ edition: 2021

#![feature(frontmatter)]

makro::check!();

// checks that a proc-macro cannot observe frontmatter tokens.
// see auxiliary/makro.rs for how it is tested.

fn main() {}
