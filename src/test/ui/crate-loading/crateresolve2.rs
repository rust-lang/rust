// check-fail

// aux-build:crateresolve2-1.rs
// aux-build:crateresolve2-2.rs
// aux-build:crateresolve2-3.rs

extern crate crateresolve2;
//~^ ERROR multiple matching crates for `crateresolve2`

fn main() {
}
