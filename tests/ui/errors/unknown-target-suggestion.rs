// Checks that an unknown --target also suggests a similar known target.
// See https://github.com/rust-lang/rust/issues/155085

// ignore-tidy-target-specific-tests
//@ compile-flags: --target x86_64-linux-gnu

fn main() {}

//~? ERROR error loading target specification: could not find specification for target "x86_64-linux-gnu"
//~? HELP run `rustc --print target-list` for a list of built-in targets
//~? HELP did you mean `x86_64-unknown-linux-gnu`
