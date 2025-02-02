//@ aux-build:remapped_dep.rs
//@ compile-flags: --remap-path-prefix={{test-suite-src-base}}/errors/auxiliary=remapped-aux

//@ revisions: local-self remapped-self
// The hack should work regardless of remapping.
//@ [remapped-self] remap-src-base

// Verify that the expected source code is shown.
//@ error-pattern: pub struct SomeStruct {} // This line should be show

extern crate remapped_dep;

fn main() {
    // The actual error is irrelevant. The important part it that is should show
    // a snippet of the dependency's source.
    let _ = remapped_dep::SomeStruct; // ~ERROR E0423
}
