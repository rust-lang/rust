// aux-build:crateresolve1-1.rs
// aux-build:crateresolve1-2.rs
// aux-build:crateresolve1-3.rs

// normalize-stderr-test: "\.nll/" -> "/"
// normalize-stderr-test: "\\\?\\" -> ""
// normalize-stderr-test: "libcrateresolve1-([123])\.[a-z]+" -> "libcrateresolve1-$1.somelib"

extern crate crateresolve1;
//~^ ERROR multiple matching crates for `crateresolve1`

fn main() {
}
