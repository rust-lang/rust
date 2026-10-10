//@ build-pass
//@ compile-flags: --crate-type staticlib,dylib -Zstaticlib-prefer-dynamic
//@ no-prefer-dynamic
//@ needs-crate-type: dylib

// Test that compiling for multiple crate types in a single compilation with
// mismatching allocator shim requirements doesn't result in the allocator shim
// missing entirely.
// In this particular test the dylib crate type will statically link libstd and
// thus need an allocator shim, while the staticlib crate type will dynamically
// link libstd and thus not need an allocator shim.
// The -Zstaticlib-prefer-dynamic flag could be avoided by doing it the other
// way around, but testing that the staticlib correctly has the allocator shim
// in that case would require a run-make test instead.

pub fn foo() {}
