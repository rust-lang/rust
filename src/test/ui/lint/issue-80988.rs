// Regression test for #80988
//
// check-pass

#![forbid(warnings)]

#[deny(warnings)]
//~^ WARNING incompatible with previous forbid
//~| WARNING being phased out
//~| WARNING incompatible with previous forbid
//~| WARNING being phased out
//~| WARNING incompatible with previous forbid
//~| WARNING being phased out
//~| WARNING incompatible with previous forbid
//~| WARNING being phased out
fn main() {}
