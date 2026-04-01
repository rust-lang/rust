//@ check-pass
//@ aux-build: issue-119369-extern.rs

// https://github.com/rust-lang/rust/pull/119369#issuecomment-1874905662

#[macro_use]
extern crate issue_119369_extern;

#[derive(Hash)]
struct A;

fn main() {
    let _: Vec<i32> = vec![];
}
