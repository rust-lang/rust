// Checking that the compiler reports multiple type errors at once

fn main() { let a: bool = 1; let b: i32 = true; }
//~^ ERROR mismatched types
//~| expected `bool`, found integer
//~| ERROR mismatched types
//~| expected `i32`, found `bool`
