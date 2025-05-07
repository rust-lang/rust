//@ run-pass

#![allow(path_statements)]
//@ proc-macro: issue-50061.rs

#![feature(decl_macro)]

extern crate issue_50061;

macro inner(any_token $v: tt) {
    $v
}

macro outer($v: tt) {
    inner!(any_token $v)
}

#[issue_50061::check]
fn main() {
    //! this doc comment forces roundtrip through a string
    let checkit = 0;
    outer!(checkit);
}
