//@ check-pass

#![warn(unused)]

#![expect(unused_variables, reason = "<This should fail and display this reason>")]
//~^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectations]
//~| NOTE `#[warn(unfulfilled_lint_expectations)]` on by default
//~| NOTE <This should fail and display this reason>

fn main() {}
