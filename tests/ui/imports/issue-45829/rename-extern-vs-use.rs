//@ aux-build:issue-45829-b.rs

mod foo {
    pub mod bar {}
}

use foo::bar;
extern crate issue_45829_b as bar;
//~^ ERROR the name `bar` is defined multiple times

fn main() {}
