//@ run-pass
//@ aux-build:issue-4545.rs


extern crate issue_4545 as somelib;
pub fn main() { somelib::mk::<isize>(); }
