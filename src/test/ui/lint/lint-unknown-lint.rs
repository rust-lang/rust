#![allow(not_a_real_lint)] //~ WARN unknown lint
#![deny(unused)]
fn main() { let unused = (); } //~ ERROR unused variable
