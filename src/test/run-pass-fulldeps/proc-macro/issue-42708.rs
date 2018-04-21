// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-42708.rs
// ignore-stage1

#![feature(decl_macro, proc_macro, proc_macro_path_invoc)]
#![allow(unused)]

extern crate issue_42708;

macro m() {
    #[derive(issue_42708::Test)]
    struct S { x: () }

    #[issue_42708::attr_test]
    struct S2 { x: () }

    #[derive(Clone)]
    struct S3 { x: () }

    fn g(s: S, s2: S2, s3: S3) {
        (s.x, s2.x, s3.x);
    }
}

m!();

fn main() {}
