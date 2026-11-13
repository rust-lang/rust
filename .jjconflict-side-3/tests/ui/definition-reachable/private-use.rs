// Check that private use statements can be used by

//@ run-pass
//@ aux-build:private-use-macro.rs

extern crate private_use_macro;

fn main() {
    assert_eq!(private_use_macro::m!(), 57);
}
