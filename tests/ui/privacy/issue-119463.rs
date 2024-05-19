//@ aux-build:issue-119463-extern.rs

extern crate issue_119463_extern;

struct S;

impl issue_119463_extern::PrivateTrait for S {
    //~^ ERROR: trait `PrivateTrait` is private
    const FOO: usize = 1;

    fn nonexistent() {}
    //~^ ERROR: method `nonexistent` is not a member of trait
}

fn main() {}
