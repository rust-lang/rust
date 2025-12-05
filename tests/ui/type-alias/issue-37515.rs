//@ check-pass

#![warn(unused)]

type Z = dyn for<'x> Send;
//~^ WARN type alias `Z` is never used

fn main() {}
