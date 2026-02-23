// While working on <https://github.com/rust-lang/rust/pull/149645>, the
// intra doc links on keyword/attribute items were not processed.

#![feature(rustdoc_internals)]
#![crate_name = "foo"]

//@ has 'foo/keyword.trait.html'
//@ has - '//a[@href="{{channel}}/core/marker/trait.Send.html"]' 'Send'
//@ has - '//a[@href="{{channel}}/core/marker/trait.Sync.html"]' 'Sync'
#[doc(keyword = "trait")]
//
/// [`Send`] and [Sync]
mod bar {}
