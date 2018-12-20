// aux-build:needs-panic-runtime.rs
// aux-build:depends.rs
// error-pattern:cannot depend on a crate that needs a panic runtime

extern crate depends;

fn main() {}
