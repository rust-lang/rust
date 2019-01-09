// aux-build:span-api-tests.rs
// aux-build:span-test-macros.rs

// ignore-pretty

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
