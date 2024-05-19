// issue: 113760

union W { s: dyn Iterator<Item = Missing> }
//~^ ERROR cannot find type `Missing` in this scope

static ONCE: W = todo!();

fn main() {}
