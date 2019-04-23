// aux-build:issue-50493.rs

#[macro_use]
extern crate issue_50493;

#[derive(Derive)] //~ ERROR field `field` of struct `Restricted` is private
struct Restricted {
    pub(in restricted) field: usize, //~ visibilities can only be restricted to ancestor modules
}

mod restricted {}

fn main() {}
