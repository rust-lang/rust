// Checking that the compiler reports multiple type errors at once

fn main() { let a: bool = 1; let b: i32 = true; }
//~^ ERROR mismatched types
//~| NOTE_NONVIRAL expected `bool`, found integer
//~| ERROR mismatched types
//~| NOTE_NONVIRAL expected `i32`, found `bool`
