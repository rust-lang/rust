// check-pass

#![feature(lint_reasons)]

#![expect(unused_mut)]
//~^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectation]

fn main() {}
