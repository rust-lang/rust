// aux-build:rustdoc-impl-parts-crosscrate.rs
// ignore-cross-compile

#![feature(negative_impls)]

extern crate rustdoc_impl_parts_crosscrate;

pub struct Bar<T> { t: T }

// The output file is html embedded in javascript, so the html tags
// aren't stripped by the processing script and we can't check for the
// full impl string.  Instead, just make sure something from each part
// is mentioned.

// @has implementors/rustdoc_impl_parts_crosscrate/trait.AnAutoTrait.js Bar
// @has - Send
// @has - !AnAutoTrait
// @has - Copy
impl<T: Send> !rustdoc_impl_parts_crosscrate::AnAutoTrait for Bar<T>
    where T: Copy {}
