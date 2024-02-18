//@ aux-build:issue-80079.rs

// using a module from another crate should not cause errors to suggest private
// items in that module

extern crate issue_80079;

use issue_80079::public;

fn main() {
    let _ = Foo; //~ ERROR cannot find value `Foo` in this scope
}
