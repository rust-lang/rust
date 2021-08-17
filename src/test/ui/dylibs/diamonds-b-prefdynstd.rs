// run-pass

// no-prefer-dynamic
// compile-flags: -C prefer-dynamic=std -Z prefer-dynamic-std

// aux-build: a_basement_both.rs

pub extern crate a_basement as a;

fn main() {
    a::a();
}
