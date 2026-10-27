#![crate_type = "lib"]
#![expect(dead_code_pub_in_binary)] //~ ERROR this lint expectation is unfulfilled
#![deny(unfulfilled_lint_expectations)]
//~^^ WARN unused attribute

pub fn unused_pub_fn() {}
