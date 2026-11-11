//@ aux-build:recursive_reexports.rs

extern crate recursive_reexports;

fn f() -> recursive_reexports::S {} //~ ERROR cannot find type `S` in crate `recursive_reexports`

fn main() {}
