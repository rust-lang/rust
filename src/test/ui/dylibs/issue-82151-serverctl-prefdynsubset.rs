// run-pass

// Make use of new `-C prefer-dynamic=...` to choose `shared` and `std` to be linked dynamically.

// no-prefer-dynamic
// compile-flags: -C prefer-dynamic=shared,std -Z prefer-dynamic-subset

// aux-build: aaa_issue_82151_bar_prefdynsubset.rs
// aux-build: aaa_issue_82151_foo_prefdynsubset.rs
// aux-build: aaa_issue_82151_shared_prefdynsubset.rs

extern crate shared;

fn main() {
    let _ = shared::Test::new();
}
