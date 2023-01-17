// aux-build:remapped_dep.rs
// compile-flags: --remap-path-prefix={{src-base}}/errors/auxiliary=remapped-aux

// The remapped paths are not normalized by compiletest.
// normalize-stderr-test: "\\(errors)" -> "/$1"

// revisions: local-self remapped-self
// [remapped-self]compile-flags: --remap-path-prefix={{src-base}}=remapped

// The paths from `remapped-self` aren't recognized by compiletest, so we
// cannot use line-specific patterns for the actual error.
// error-pattern: E0423

// Verify that the expected source code is shown.
// error-pattern: pub struct SomeStruct {} // This line should be show

extern crate remapped_dep;

fn main() {
    // The actual error is irrelevant. The important part it that is should show
    // a snippet of the dependency's source.
    let _ = remapped_dep::SomeStruct;
}
