#![feature(cfg_accessible)]

#[cfg_accessible(Z)] //~ ERROR cannot determine whether the path is accessible or not
struct S;

#[cfg_accessible(S)] //~ ERROR cannot determine whether the path is accessible or not
struct Z;

fn main() {}
