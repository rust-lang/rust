// Regression test for an ICE: https://github.com/rust-lang/rust/issues/120884
// We still need to properly go through coercions between types with errors instead of
// shortcutting and returning success, because we need the adjustments for building the MIR.

pub fn has_error() -> TypeError {}
//~^ ERROR cannot find type `TypeError` in this scope

pub fn cast() -> *const u8 {
    // Casting a function item to a data pointer in valid in HIR, but invalid in MIR.
    // We need an adjustment (ReifyFnPointer) to insert a cast from the function item
    // to a function pointer as a separate MIR statement.
    has_error as *const u8
}

fn main() {}
