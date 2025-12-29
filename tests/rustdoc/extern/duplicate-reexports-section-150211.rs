//@ aux-build:pub-extern-crate.rs

// Regression test for issue <https://github.com/rust-lang/rust/issues/150211>.
// When a module has both `pub extern crate` and `pub use` items,
// they should both appear under a single "Re-exports" section,
// not two separate sections.

//@ has duplicate_reexports_section_150211/index.html
// Verify there's exactly one Re-exports section header
//@ count - '//h2[@id="reexports"]' 1
//@ has - '//h2[@id="reexports"]' 'Re-exports'
// Verify both the extern crate and the use item are present
//@ has - '//code' 'pub extern crate inner;'
//@ has - '//code' 'pub use inner::SomeStruct;'

pub extern crate inner;

#[doc(no_inline)]
pub use inner::SomeStruct;
