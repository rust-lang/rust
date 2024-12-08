// https://github.com/rust-lang/rust/issues/42760
#![crate_name="foo"]

#![allow(rustdoc::invalid_rust_codeblocks)]

//@ has foo/struct.NonGen.html
//@ has - '//h2' 'Example'

/// Item docs.
///
#[doc="Hello there!"]
///
/// # Example
///
/// ```rust
/// // some code here
/// ```
pub struct NonGen;
