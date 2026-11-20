//@ compile-flags: --extern pub_struct
//@ aux-build:pub-struct.rs

/// [SomeStruct]
///
/// [SomeStruct]: pub_struct::SomeStruct
pub fn foo() {}
