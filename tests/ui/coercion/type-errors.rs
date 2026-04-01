// Test that we don't ICE for coercions with type errors.
// We still need to properly go through coercions between types with errors instead of
// shortcutting and returning success, because we need the adjustments for building the MIR.

pub fn has_error() -> TypeError {}
//~^ ERROR cannot find type `TypeError` in this scope

// https://github.com/rust-lang/rust/issues/120884
// Casting a function item to a data pointer in valid in HIR, but invalid in MIR.
// We need an adjustment (ReifyFnPointer) to insert a cast from the function item
// to a function pointer as a separate MIR statement.
pub fn cast() -> *const u8 {
    has_error as *const u8
}

// https://github.com/rust-lang/rust/issues/120945
// This one ICEd, because we skipped the builtin deref from `&TypeError` to `TypeError`.
pub fn autoderef_source(e: &TypeError) {
    //~^ ERROR cannot find type `TypeError` in this scope
    autoderef_target(e)
}

pub fn autoderef_target(_: &TypeError) {}
//~^ ERROR cannot find type `TypeError` in this scope

fn main() {}
