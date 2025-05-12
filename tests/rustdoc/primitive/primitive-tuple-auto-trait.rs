//@ compile-flags: --crate-type lib
//@ edition: 2018

#![crate_name = "foo"]
#![feature(rustc_attrs)]

//@ has foo/primitive.tuple.html '//h1' 'Primitive Type tuple'
//@ has - '//section[@id="main-content"]//div[@class="docblock"]//p' 'this is a test!'
//@ has - '//h2[@id="synthetic-implementations"]' 'Auto Trait Implementations'
//@ has - '//div[@id="synthetic-implementations-list"]//h3' 'Send'
//@ has - '//div[@id="synthetic-implementations-list"]//h3' 'Sync'
#[rustc_doc_primitive = "tuple"]
/// this is a test!
///
// Hardcoded anchor to header written in library/core/src/primitive_docs.rs
//@ has - '//h2[@id="trait-implementations-1"]' 'Trait implementations'
/// # Trait implementations
///
/// This header is hard-coded in the HTML format linking for `#[doc(fake_variadics)]`.
/// To make sure it gets linked correctly, we need to make sure the hardcoded anchor
/// in the code matches what rustdoc generates for the header.
mod tuple_prim {}
