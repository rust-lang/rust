// Check that functions visible to macros through paths with >2 segments are
// considered reachable

//@ aux-build:field-method-macro.rs
//@ run-pass

extern crate field_method_macro;

fn main() {
    assert_eq!(field_method_macro::m!(), 33);
}
