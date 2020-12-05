// check-pass

#![warn(unused)]

type Z = dyn for<'x> Send;
//~^ WARN type alias is never used

fn main() {}
