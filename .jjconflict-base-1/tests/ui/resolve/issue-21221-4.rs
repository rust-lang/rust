// testing whether the lookup mechanism picks up types
// defined in the outside crate

//@ aux-build:issue-21221-4.rs

extern crate issue_21221_4;

struct Foo;

impl T for Foo {}
//~^ ERROR cannot find trait `T`

fn main() {
    println!("Hello, world!");
}
