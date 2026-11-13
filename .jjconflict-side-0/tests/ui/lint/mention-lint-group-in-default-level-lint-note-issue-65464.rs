//@ check-pass

// Verify information about membership to builtin lint group is included in the lint message when
// explaining lint level and source for builtin lints with default settings.
//
// Ideally, we'd like to use lints that are part of `unused` group as shown in the issue.
// This is not possible in a ui test, because `unused` lints are enabled with `-A unused`
// in such tests, and the we're testing a scenario with no modification to the default settings.

fn main() {
    // additional context is provided only if the level is not explicitly set
    let WrongCase = 1;
    //~^ WARN [non_snake_case]
    //~| NOTE `#[warn(non_snake_case)]` (part of `#[warn(nonstandard_style)]`) on by default

    // unchanged message if the level is explicitly set
    // even if the level is the same as the default
    #[warn(nonstandard_style)] //~ NOTE the lint level is defined here
    let WrongCase = 2;
    //~^ WARN [non_snake_case]
    //~| NOTE `#[warn(non_snake_case)]` implied by `#[warn(nonstandard_style)]`
}
