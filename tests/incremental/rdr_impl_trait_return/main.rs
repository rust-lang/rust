// Test impl Trait return types with private concrete types.
// Known limitation: changes to the private concrete type behind `impl Trait`
// currently DO cause downstream rebuilds, as the concrete type leaks through
// the opaque return type.
//
// - rpass1: Initial compilation
// - rpass2: Private struct gets extra field
// - rpass3: Another private struct added

//@ revisions: rpass1 rpass2 rpass3
//@ aux-build: impl_trait_dep.rs
//@ edition: 2024
//@ ignore-backends: gcc

extern crate impl_trait_dep;

use impl_trait_dep::MyTrait;

fn main() {
    let thing = impl_trait_dep::make_thing();
    assert_eq!(thing.value(), 42);
}
