// issue: 113760

union W { s: dyn Iterator<Item = Missing> }
//~^ ERROR cannot find type `Missing` in this scope
//~| ERROR field must implement `Copy` or be wrapped in `ManuallyDrop<...>` to be used in a union

static ONCE: W = todo!();

fn main() {}
