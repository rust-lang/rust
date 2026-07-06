//@ check-pass

// Regression test for https://github.com/rust-lang/rust/issues/158745.
// This checks that rustdoc resolves intra-doc links in deprecation notes using
// the re-export that carries the attribute, even when that re-export is later
// inlined through another re-export.

#![deny(rustdoc::broken_intra_doc_links)]

mod bar {
    pub struct A;
}

#[deprecated(note = "[std::io::ErrorKind::NotFound]")]
pub use bar::A;

#[doc(inline)]
pub use A as X;
