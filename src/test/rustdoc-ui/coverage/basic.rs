// compile-flags:-Z unstable-options --show-coverage
// check-pass

#![feature(extern_types)]

//! Make sure to have some docs on your crate root

/// This struct is documented, but its fields are not.
///
/// However, one field is private, so it shouldn't show in the total.
pub struct SomeStruct {
    pub some_field: usize,
    other_field: usize,
}

impl SomeStruct {
    /// Method with docs
    pub fn this_fn(&self) {}

    // Method without docs
    pub fn other_method(&self) {}
}

// struct without docs
pub struct OtherStruct;

// function with no docs
pub fn some_fn() {}

/// Function with docs
pub fn other_fn() {}

pub enum SomeEnum {
    /// Some of these variants are documented...
    VarOne,
    /// ...but some of them are not.
    VarTwo,
    // (like this one)
    VarThree,
}

/// There's a macro here, too
#[macro_export]
macro_rules! some_macro {
    () => {};
}

extern "C" {
    pub type ExternType;
}
