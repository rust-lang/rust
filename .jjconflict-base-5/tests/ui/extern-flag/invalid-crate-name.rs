//@ compile-flags: --extern=?#1%$

fn main() {}

//~? ERROR crate name `?#1%$` passed to `--extern` is not a valid ASCII identifier
