// run-pass
// aux-build:variants.rs

extern crate variants;

use variants::*;

fn main() {
    check_variants();

    test_variants!();
    test_variants2!();

    assert_eq!(MyEnum::Variant as u8, 1);
}
