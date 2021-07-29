// check-pass

#![feature(lint_reasons)]

#![expect(unused_variables, reason = "<This should fail and display this reason>")]
//~^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectation]
//~| NOTE `#[warn(unfulfilled_lint_expectation)]` on by default
//~| NOTE <This should fail and display this reason>

fn main() {}
