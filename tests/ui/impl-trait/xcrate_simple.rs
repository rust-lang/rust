// run-pass

// aux-build:xcrate.rs

extern crate xcrate;

fn main() {
    xcrate::return_internal_fn()();
}
