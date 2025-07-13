//@ run-pass
#![allow(unused_variables)]
//@ aux-build:issue-7899.rs


extern crate issue_7899 as testcrate;

fn main() {
    let f = testcrate::V2(1.0f32, 2.0f32);
}

// https://github.com/rust-lang/rust/issues/7899
