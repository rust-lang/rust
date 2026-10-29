// testing whether the lookup mechanism picks up types
// defined in the outside crate

#![crate_type="lib"]

mod foo {
    // should not be suggested => foo is private
    pub trait T {}
}

// should be suggested
pub use foo::T;
