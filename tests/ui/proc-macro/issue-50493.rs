//@ proc-macro: issue-50493.rs

#[macro_use]
extern crate issue_50493;

#[derive(Derive)]
struct Restricted {
    pub(in crate::restricted) field: usize, //~ ERROR visibilities can only be restricted to ancestor modules
}

mod restricted {}

fn main() {}
