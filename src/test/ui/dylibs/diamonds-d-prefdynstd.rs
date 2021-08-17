// run-pass

// Make use of new `-C prefer-dynamic=...` flag to allow *only* `std` to be linked dynamically.

// no-prefer-dynamic
// compile-flags: -C prefer-dynamic=std -Z prefer-dynamic-std -la_basement

// aux-build: a_basement_dynamic.rs

extern crate a_basement as a;

fn main() {
    a::a();
}
