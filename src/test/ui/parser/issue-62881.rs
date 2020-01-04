fn main() {}

fn f() -> isize { fn f() -> isize {} pub f<
//~^ ERROR missing `fn` or `struct` for function or struct definition
//~| ERROR mismatched types
//~ ERROR this file contains an unclosed delimiter
