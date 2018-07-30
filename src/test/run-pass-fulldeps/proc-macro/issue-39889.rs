// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-39889.rs
// ignore-stage1

#![feature(use_extern_macros)]
#![allow(unused)]

extern crate issue_39889;
use issue_39889::Issue39889;

#[derive(Issue39889)]
struct S;

fn main() {}
