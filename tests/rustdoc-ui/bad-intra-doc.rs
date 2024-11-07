#![no_std]
#![deny(rustdoc::broken_intra_doc_links)]

// regression test for https://github.com/rust-lang/rust/issues/54191

/// this is not a link to [`example.com`]
///
/// this link [`has spaces in it`].
///
/// attempted link to method: [`Foo.bar()`]
///
/// classic broken intra-doc link: [`Bar`]
pub struct Foo;
