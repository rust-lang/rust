#![feature(cfg_accessible)]

#[cfg_accessible(Z)] // OK, recovered after the other `cfg_accessible` produces an error.
struct S;

#[cfg_accessible(S)] //~ ERROR cannot determine whether the path is accessible or not
struct Z;

fn main() {}
