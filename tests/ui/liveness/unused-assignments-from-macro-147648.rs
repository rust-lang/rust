//@ check-pass
//@ proc-macro: aux_issue_147648.rs

#![deny(unused_assignments)]
#![allow(dead_code)]
#![allow(unused_variables)]

extern crate aux_issue_147648;
use aux_issue_147648::UnusedAssign;

#[derive(UnusedAssign)]
pub struct MyError {
    source_code: (),
}

fn main() {
    let _error = MyError { source_code: () };
}
