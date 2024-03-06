//@ aux-build:issue-45829-b.rs

extern crate issue_45829_b;
use std as issue_45829_b;
//~^ ERROR is defined multiple times

fn main() {}
