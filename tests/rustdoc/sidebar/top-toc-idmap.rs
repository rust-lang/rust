#![crate_name = "foo"]
#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

//! # Structs
//!
//! This header has the same name as a built-in header,
//! and we need to make sure they're disambiguated with
//! suffixes.
//!
//! Module-like headers get derived from the internal ID map,
//! so the *internal* one gets a suffix here. To make sure it
//! works right, the one in the `top-toc` needs to match the one
//! in the `top-doc`, and the one that's not in the `top-doc`
//! needs to match the one that isn't in the `top-toc`.

//@ has foo/index.html
// User header
//@ has - '//section[@id="rustdoc-toc"]/ul[@class="block top-toc"]/li/a[@href="#structs"]' 'Structs'
//@ has - '//details[@class="toggle top-doc"]/div[@class="docblock"]/h2[@id="structs"]' 'Structs'
// Built-in header
//@ has - '//section[@id="rustdoc-toc"]/ul[@class="block"]/li/a[@href="#structs-1"]' 'Structs'
//@ has - '//section[@id="main-content"]/h2[@id="structs-1"]' 'Structs'

/// # Fields
/// ## Fields
/// ### Fields
///
/// The difference between struct-like headers and module-like headers
/// is strange, but not actually a problem as long as we're consistent.

//@ has foo/struct.MyStruct.html
// User header
//@ has - '//section[@id="rustdoc-toc"]/ul[@class="block top-toc"]/li/a[@href="#fields-1"]' 'Fields'
//@ has - '//details[@class="toggle top-doc"]/div[@class="docblock"]/h2[@id="fields-1"]' 'Fields'
// Only one level of nesting
//@ count - '//section[@id="rustdoc-toc"]/ul[@class="block top-toc"]//a' 2
// Built-in header
//@ has - '//section[@id="rustdoc-toc"]/h3/a[@href="#fields"]' 'Fields'
//@ has - '//section[@id="main-content"]/h2[@id="fields"]' 'Fields'

pub struct MyStruct {
    pub fields: i32,
}
