// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:span-api-tests.rs
// aux-build:span-test-macros.rs

// ignore-pretty

#![feature(use_extern_macros)]

#[macro_use]
extern crate span_test_macros;

extern crate span_api_tests;

use span_api_tests::{reemit, assert_fake_source_file, assert_source_file};

macro_rules! say_hello {
    ($macname:ident) => ( $macname! { "Hello, world!" })
}

assert_source_file! { "Hello, world!" }

say_hello! { assert_source_file }

reemit_legacy! {
    assert_source_file! { "Hello, world!" }
}

say_hello_extern! { assert_fake_source_file }

reemit! {
    assert_source_file! { "Hello, world!" }
}

fn main() {}
