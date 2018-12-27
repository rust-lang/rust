// aux-build:issue-41211.rs

// FIXME: https://github.com/rust-lang/rust/issues/41430
// This is a temporary regression test for the ICE reported in #41211

#![feature(custom_inner_attributes)]

#![emit_unchanged]
//~^ ERROR attribute `emit_unchanged` is currently unknown to the compiler
extern crate issue_41211;
use issue_41211::emit_unchanged;

fn main() {}
