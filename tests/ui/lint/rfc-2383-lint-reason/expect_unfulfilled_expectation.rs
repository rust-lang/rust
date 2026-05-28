//@ check-pass
// ignore-tidy-linelength

#![warn(unused_mut)]

#![expect(unfulfilled_lint_expectations, reason = "idk why you would expect this")]
//~^ WARNING this lint expectation is unfulfilled
//~| NOTE `#[warn(unfulfilled_lint_expectations)]` on by default
//~| NOTE idk why you would expect this
//~| NOTE the `unfulfilled_lint_expectations` lint can't be expected and will always produce this message

#[expect(unfulfilled_lint_expectations, reason = "a local: idk why you would expect this")]
//~^ WARNING this lint expectation is unfulfilled
//~| NOTE a local: idk why you would expect this
//~| NOTE the `unfulfilled_lint_expectations` lint can't be expected and will always produce this message
pub fn normal_test_fn() {
    #[expect(unused_mut, reason = "this expectation will create a diagnostic with the default lint level")]
    //~^ WARNING this lint expectation is unfulfilled
    //~| WARNING this lint expectation is unfulfilled
    //~| NOTE this expectation will create a diagnostic with the default lint level
    //~| NOTE this expectation will create a diagnostic with the default lint level
    //~| NOTE duplicate diagnostic emitted due to `-Z deduplicate-diagnostics=no`
    let mut v = vec![1, 1, 2, 3, 5];
    v.sort();

    // Check that lint lists including `unfulfilled_lint_expectations` are also handled correctly
    #[expect(unused, unfulfilled_lint_expectations, reason = "the expectation for `unused` should be fulfilled")]
    //~^ WARNING this lint expectation is unfulfilled
    //~| WARNING this lint expectation is unfulfilled
    //~| NOTE the expectation for `unused` should be fulfilled
    //~| NOTE the expectation for `unused` should be fulfilled
    //~| NOTE the `unfulfilled_lint_expectations` lint can't be expected and will always produce this message
    //~| NOTE the `unfulfilled_lint_expectations` lint can't be expected and will always produce this message
    //~| NOTE duplicate diagnostic emitted due to `-Z deduplicate-diagnostics=no`
    let value = "I'm unused";
}

#[expect(warnings, reason = "this suppresses all warnings and also suppresses itself. No warning will be issued")]
pub fn expect_warnings() {
    // This lint trigger will be suppressed
    #[warn(unused_mut)]
    let mut v = vec![1, 1, 2, 3, 5];
}

fn main() {}
