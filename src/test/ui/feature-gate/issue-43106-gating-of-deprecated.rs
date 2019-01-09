// This test just shows that a crate-level `#![deprecated]` does not
// signal a warning or error. (This file sits on its own because a
// crate-level `#![deprecated]` causes all that crate's item
// definitions to be deprecated, which is a pain to work with.)
//
// (For non-crate-level cases, see issue-43106-gating-of-builtin-attrs.rs)

// compile-pass
// skip-codegen
#![allow(dead_code)]
#![deprecated           = "1100"]

// Since we expect for the mix of attributes used here to compile
// successfully, and we are just testing for the expected warnings of
// various (mis)uses of attributes, we use the `rustc_error` attribute
// on the `fn main()`.


fn main() {
    println!("Hello World");
}
