#![feature(optin_builtin_traits)]

use std::marker::Send;

struct TestType;

unsafe impl !Send for TestType {}
//~^ ERROR E0198

fn main() {}
