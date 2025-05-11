// Checking that the compiler reports multiple type errors at once

//@ dont-require-annotations: NOTE

fn main() { let a: bool = 1; let b: i32 = true; }
//~^ ERROR mismatched types
//~| NOTE expected `bool`, found integer
//~| ERROR mismatched types
//~| NOTE expected `i32`, found `bool`
