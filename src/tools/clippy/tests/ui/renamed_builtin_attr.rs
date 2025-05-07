//@compile-flags: -Zdeduplicate-diagnostics=yes

#[clippy::cyclomatic_complexity = "1"]
//~^ ERROR: usage of deprecated attribute
fn main() {}
