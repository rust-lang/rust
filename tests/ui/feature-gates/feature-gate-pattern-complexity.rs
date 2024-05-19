// check that `pattern_complexity` is feature-gated

#![pattern_complexity = "42"]
//~^ ERROR: the `#[pattern_complexity]` attribute is just used for rustc unit tests

fn main() {}
