// This test ensures that the `doc_cfg` feature works on foreign types impl.
// Regression test for <https://github.com/rust-lang/rust/issues/150268>.

// ignore-tidy-linelength

#![feature(doc_cfg)]
#![crate_name = "foo"]

//@has 'foo/trait.Blob.html'
//@has - '//*[@id="impl-Blob-for-Box%3CR%3E"]//*[@class="stab portability"]' 'Available on non-crate feature alloc only.'

pub trait Blob {}

#[cfg(not(feature = "alloc"))]
impl<R: ?Sized> Blob for std::boxed::Box<R> {}
