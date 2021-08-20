use std::fmt::Debug;

// check-pass

fn in_adt_in_return() -> Vec<impl Debug> { panic!() }

fn main() {}
