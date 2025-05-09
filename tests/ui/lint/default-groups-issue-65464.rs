//@ check-pass

// Verify information about membership to builtin lint group is included in the lint message when
// explaining lint level and source for builtin lints with default settings.
//
// Ideally, we'd like to use lints that are part of `unused` group as shown in the issue.
// This is not possible in a ui test, because `unused` lints are enabled with `-A unused`
// in such tests, and the we're testing a scenario with no modification to the default settings.

fn main() {
    let WrongCase = 1;
    //~^ WARN [non_snake_case]
    //~| NOTE `#[warn(non_snake_case)]` (part of `#[warn(nonstandard_style)]`) on by default
}
