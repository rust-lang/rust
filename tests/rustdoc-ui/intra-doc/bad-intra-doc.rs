#![no_std]
#![deny(rustdoc::broken_intra_doc_links)]

// regression test for https://github.com/rust-lang/rust/issues/54191

/// this is not a link to [`example.com`] //~ERROR unresolved link
///
/// this link [`has spaces in it`].
///
/// attempted link to method: [`Foo.bar()`] //~ERROR unresolved link
///
/// classic broken intra-doc link: [`Bar`] //~ERROR unresolved link
///
/// no backticks, so we let this one slide: [Foo.bar()]
pub struct Foo;
