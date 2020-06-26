#![feature(negative_impls)]

use std::marker::Send;

struct TestType;

unsafe impl !Send for TestType {}
//~^ ERROR E0198

fn main() {}
