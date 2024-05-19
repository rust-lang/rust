//@ run-pass
#![allow(unused_variables)]
//@ aux-build:issue-25467.rs

pub type Issue25467BarT = ();
pub type Issue25467FooT = ();

extern crate issue_25467 as aux;

fn main() {
    let o: aux::Object = None;
}
