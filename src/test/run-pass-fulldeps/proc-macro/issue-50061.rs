// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-50061.rs
// ignore-stage1

#![feature(proc_macro, proc_macro_path_invoc, decl_macro)]

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
