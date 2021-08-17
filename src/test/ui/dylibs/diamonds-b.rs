// build-fail

// The a_basement dependency has been compiled with `-C prefer-dynamic=no`,
// which ends up leading to a duplication of `std` (and all crates underneath
// it) in both the `a_basement` crate and in this crate. Rust does not support
// having duplicate libraries like that, so this compilation will fail.

// aux-build: a_basement_both.rs

pub extern crate a_basement as a;

fn main() {
    a::a();
}
