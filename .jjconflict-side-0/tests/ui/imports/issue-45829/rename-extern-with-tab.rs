//@ aux-build:issue-45829-a.rs
//@ aux-build:issue-45829-b.rs

extern crate issue_45829_a;
extern  crate    issue_45829_b  as  issue_45829_a;
//~^ ERROR is defined multiple times

fn main() {}
