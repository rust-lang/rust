// Make sure that macro expanded codegen attributes work across crates.
// We used to gensym the identifiers in attributes, which stopped dependent
// crates from seeing them, resulting in linker errors in cases like this one.

//@ run-pass
//@ aux-build:codegen-attrs.rs

extern crate codegen_attrs;

fn main() {
    assert_eq!(codegen_attrs::rust_function_name(), 2);
}
