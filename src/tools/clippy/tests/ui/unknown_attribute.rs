//@compile-flags: -Zdeduplicate-diagnostics=yes

#[clippy::unknown]
//~^ ERROR: usage of unknown attribute
#[clippy::cognitive_complexity = "1"]
fn main() {}
