//@ check-pass

#![crate_type = "bin"]
#![expect(dead_code_pub_in_binary)]
#![deny(unfulfilled_lint_expectations)]

pub fn unused_pub_fn() {}

fn main() {}
