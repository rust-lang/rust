//~ ERROR extern location for `std` does not exist at ``
//@ compile-flags: --extern std=
//@ needs-unwind since it affects the error output

fn main() {}
