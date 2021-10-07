// aux-build:crateresolve1-1.rs
// aux-build:crateresolve1-2.rs
// aux-build:crateresolve1-3.rs

// normalize-stderr-test: "\.so" -> ".dylib"
// normalize-stderr-test: "\.dll" -> ".dylib"

extern crate crateresolve1;
//~^ ERROR multiple matching crates for `crateresolve1`

fn main() {
}
