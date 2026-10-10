// Test that variants of an enum defined in another crate are resolved
// correctly when their names differ only in `SyntaxContext`.

//@ run-pass
//@ aux-build:variants.rs

extern crate variants;

use variants::*;

fn main() {
    check_variants();

    test_variants!();
    test_variants2!();

    assert_eq!(MyEnum::Variant as u8, 1);
}
