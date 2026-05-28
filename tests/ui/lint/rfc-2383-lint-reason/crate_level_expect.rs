//@ check-pass

#![warn(unused)]

#![expect(unused_mut)]
//~^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectations]
//~| NOTE `#[warn(unfulfilled_lint_expectations)]` on by default

#![expect(unused_variables)]

fn main() {
    let x = 0;
}
