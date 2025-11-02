//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows
//@normalize-stderr-test: "\n *= note: inside `std::.*" -> ""

// A lot of code runs before main, which we should be able to handle in GenMC mode.

fn main() {}
