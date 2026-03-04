//@ check-pass

#![warn(unused)]

// The warnings are not double triggers, they identify different unfulfilled lint
// expectations one for each listed lint.

#[expect(unused_variables, unused_mut, while_true)]
//~^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectations]
//~| NOTE `#[warn(unfulfilled_lint_expectations)]` on by default
//~| WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectations]
fn check_multiple_lints_1() {
    // This only trigger `unused_variables`
    let who_am_i = 666;
}

#[expect(unused_variables, unused_mut, while_true)]
//~^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectations]
//~| WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectations]
fn check_multiple_lints_2() {
    // This only triggers `unused_mut`
    let mut x = 0;
    println!("I use x: {}", x);
}

#[expect(unused_variables, unused_mut, while_true)]
//~^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectations]
//~| WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectations]
fn check_multiple_lints_3() {
    // This only triggers `while_true` which is also an early lint
    while true {}
}

#[expect(unused, while_true)]
//~^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectations]
fn check_multiple_lints_with_lint_group_1() {
    let who_am_i = 666;

    let mut x = 0;
    println!("I use x: {}", x);
}

#[expect(unused, while_true)]
//~^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectations]
fn check_multiple_lints_with_lint_group_2() {
    while true {}
}

fn main() {
    check_multiple_lints_1();
    check_multiple_lints_2();
    check_multiple_lints_3();

    check_multiple_lints_with_lint_group_1();
    check_multiple_lints_with_lint_group_2();
}
