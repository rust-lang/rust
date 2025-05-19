//! Test file for missing_docs_in_private_items lint with allow_unused configuration
#![warn(clippy::missing_docs_in_private_items)]
#![allow(dead_code)]

/// A struct with some documented and undocumented fields
struct Test {
    /// This field is documented
    field1: i32,
    _unused: i32, // This should not trigger a warning because it starts with an underscore
    field3: i32,  //~ missing_docs_in_private_items
}

struct Test2 {
    //~^ missing_docs_in_private_items
    _field1: i32, // This should not trigger a warning
    _field2: i32, // This should not trigger a warning
}

struct Test3 {
    //~^ missing_docs_in_private_items
    /// This field is documented although this is not mandatory
    _unused: i32, // This should not trigger a warning because it starts with an underscore
    field2: i32, //~ missing_docs_in_private_items
}

fn main() {}
