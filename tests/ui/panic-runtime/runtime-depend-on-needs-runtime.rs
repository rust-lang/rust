// dont-check-compiler-stderr
//@aux-build:needs-panic-runtime.rs
//@aux-build:depends.rs
//@error-in-other-file:cannot depend on a crate that needs a panic runtime

extern crate depends;

fn main() {}
