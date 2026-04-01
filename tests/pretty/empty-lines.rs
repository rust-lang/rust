//@ compile-flags: --crate-type=lib

// Issue #759
// Whitespace under block opening should not expand forever

fn a() -> usize {

    1
}
