// aux-build:issue-50493.rs

#[macro_use]
extern crate issue_50493;

#[derive(Derive)]
struct Restricted {
    pub(in restricted) field: usize, //~ visibilities can only be restricted to ancestor modules
}

mod restricted {}

fn main() {}
