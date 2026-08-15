//! Tests that arrays and slices in constants aren't interchangeable when used as patterns.

#[derive(PartialEq, Eq)]
struct SomeStruct<T: ?Sized>(T);

const BSTR_SIZED: &'static [u8; 3] = b"012";
const BSTR_UNSIZED: &'static [u8] = BSTR_SIZED;
const STRUCT_SIZED: &'static SomeStruct<[u8; 3]> = &SomeStruct(*BSTR_SIZED);
const STRUCT_UNSIZED: &'static SomeStruct<[u8]> = STRUCT_SIZED;

fn type_mismatches() {
    // Test that array consts can't be used where a slice pattern is expected. This helps ensure
    // that `const_to_pat` won't produce irrefutable `thir::PatKind::Array` patterns when matching
    // on slices, which would result in missing length checks.
    // See also `tests/ui/match/pattern-deref-miscompile.rs`, which tests that byte string literal
    // patterns check slices' length appropriately when matching on slices.
    match BSTR_UNSIZED {
        BSTR_SIZED => {}
        //~^ ERROR: mismatched types
        _ => {}
    }
    match STRUCT_UNSIZED {
        STRUCT_SIZED => {}
        //~^ ERROR: mismatched types
        _ => {}
    }

    // Test that slice consts can't be used where an array pattern is expected.
    match BSTR_UNSIZED {
        BSTR_SIZED => {}
        //~^ ERROR: mismatched types
        _ => {}
    }
    // If the types matched here, this would still error, since unsized structs aren't permitted in
    // constant patterns. See the `invalid_patterns` test below.
    match STRUCT_UNSIZED {
        STRUCT_SIZED => {}
        //~^ ERROR: mismatched types
        _ => {}
    }
}

fn invalid_patterns() {
    // Test that unsized structs containing slices can't be used as patterns.
    // See `tests/ui/consts/issue-87046.rs` for an example with `str`.
    match STRUCT_UNSIZED {
        STRUCT_UNSIZED => {}
        //~^ ERROR: cannot use unsized non-slice type `SomeStruct<[u8]>` in constant patterns
        _ => {}
    }
}

fn main() {}
