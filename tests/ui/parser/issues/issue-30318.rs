// run-rustfix
#![allow(unused)]
fn foo() { }

//! Misplaced comment...
//~^ ERROR expected outer doc comment
fn bar() { } //~ NOTE the inner doc comment doesn't annotate this function

#![test] //~ ERROR an inner attribute is not permitted in this context
fn baz() { } //~ NOTE the inner attribute doesn't annotate this function
//~^^ NOTE inner attributes, like `#![no_std]`, annotate the item enclosing them, and are usually

/*! Misplaced comment... */
//~^ ERROR expected outer doc comment
fn bat() { } //~ NOTE the inner doc comment doesn't annotate this function

fn main() { }

//! Misplaced comment...
//~^ ERROR expected outer doc comment
//~| NOTE inner doc comments like this (starting with `//!` or `/*!`) can only appear before items
//~| NOTE other attributes here
/*! Misplaced comment... */
//~^ ERROR expected outer doc comment
//~| NOTE this doc comment doesn't document anything
//~| ERROR expected item after doc comment
//~| NOTE inner doc comments like this (starting with `//!` or `/*!`) can only appear before items
