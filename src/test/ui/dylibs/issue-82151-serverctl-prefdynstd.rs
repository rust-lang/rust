// run-pass

// Make use of new `-C prefer-dynamic=...` flag to allow *only* `std` to be
// linked dynamically via the rustc injected flags, and then also manually link
// to `shared`.

// no-prefer-dynamic
// compile-flags: -C prefer-dynamic=std -Z prefer-dynamic-std -lshared

// aux-build: aaa_issue_82151_bar_prefdynstd.rs
// aux-build: aaa_issue_82151_foo_prefdynstd.rs
// aux-build: aaa_issue_82151_shared_prefdynstd.rs

extern crate shared;

fn main() {
    let _ = shared::Test::new();
}
