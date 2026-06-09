// check that `pattern_complexity_limit` is feature-gated

#![pattern_complexity_limit = "42"]
//~^ ERROR: use of an internal attribute [E0658]
//~| NOTE the `#[pattern_complexity_limit]` attribute is an internal implementation detail that will never be stable
//~| NOTE: the `#[pattern_complexity_limit]` attribute is used for rustc unit tests

fn main() {}
