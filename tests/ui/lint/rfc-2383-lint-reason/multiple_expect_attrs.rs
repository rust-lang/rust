//@ check-pass

#![warn(unused)]

#[warn(unused_variables)]
#[expect(unused_variables)]
//~^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectations]
//~| NOTE `#[warn(unfulfilled_lint_expectations)]` on by default
#[allow(unused_variables)]
#[expect(unused_variables)] // Only this expectation will be fulfilled
fn main() {
    let x = 2;
}
