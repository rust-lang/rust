//@ compile-flags: --test --crate-type=lib

// `--test` forces the crate type to `Executable`, but an explicit `--crate-type=lib`
// on the command line means this is a library's unit-test harness, so the lint
// must not fire.
#![expect(dead_code_pub_in_binary)] //~ ERROR this lint expectation is unfulfilled
#![deny(unfulfilled_lint_expectations)]
//~^^ WARN unused attribute

pub fn unused_pub_fn() {}
