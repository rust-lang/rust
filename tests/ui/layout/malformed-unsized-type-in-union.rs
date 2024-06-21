// issue: 113760

union W { s: dyn Iterator<Item = Missing> }
//~^ ERROR cannot find type `Missing`

static ONCE: W = todo!();

fn main() {}
