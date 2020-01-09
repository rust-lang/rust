// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

#![feature(negative_impls)]

use std::marker::Send;

struct TestType;

impl !Send for TestType {}

fn main() {}
