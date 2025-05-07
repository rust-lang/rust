// This test ensures that an empty associated item will not crash rustdoc.
// This is a regression test for <https://github.com/rust-lang/rust/issues/140026>.

#[deny(rustdoc::broken_intra_doc_links)]

/// [`String::`]
//~^ ERROR
pub struct Foo;
