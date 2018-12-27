// aux-build:derive-a-b.rs

#[macro_use]
extern crate derive_a_b;

fn main() {
    // Test that constructing the `visible_parent_map` (in `cstore_impl.rs`) does not ICE.
    std::cell::Cell::new(0) //~ ERROR mismatched types
}
