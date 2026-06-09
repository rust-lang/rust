// Check what happens when we forbid a bigger group but
// then deny a subset of that group.

#![forbid(warnings)]
#![deny(forbidden_lint_groups)]

#[allow(nonstandard_style)]
//~^ ERROR incompatible with previous
//~| WARNING previously accepted by the compiler
//~| ERROR incompatible with previous
//~| WARNING previously accepted by the compiler
//~| ERROR incompatible with previous
//~| WARNING previously accepted by the compiler
fn main() {}
