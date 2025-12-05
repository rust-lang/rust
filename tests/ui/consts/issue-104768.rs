const A: &_ = 0_u32;
//~^ ERROR: the placeholder `_` is not allowed within types on item signatures for constants
//~| ERROR: mismatched types

fn main() {}
