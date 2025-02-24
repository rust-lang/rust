// check that `pattern_complexity_limit` is feature-gated

#![pattern_complexity_limit = "42"]
//~^ ERROR: the `#[pattern_complexity_limit]` attribute is just used for rustc unit tests

fn main() {}
