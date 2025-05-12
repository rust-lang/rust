// Tests for the issue in #137589
#[crate_type = foo!()]
//~^ ERROR cannot find macro `foo` in this scope

macro_rules! foo {} //~ ERROR unexpected end of macro invocation

fn main() {}
