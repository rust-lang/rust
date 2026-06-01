// Regression test for <https://github.com/rust-lang/rust/issues/157210>.
// Rustdoc used to ICE when a `pub use` inside a private module had a
// `#[deprecated]` attribute with an intra-doc link in its note.

#![deny(rustdoc::broken_intra_doc_links)]

// Private module: items are not exported so no link warning is expected,
// but rustdoc must not ICE either.
mod inner {
    #[deprecated = "[A](TypeAlias::x)"]
    pub use std::vec::Vec;
}

// Exported re-export: the broken intra-doc link should produce a warning.
#[deprecated = "[A](TypeAlias::x)"]
//~^ ERROR: unresolved link to `TypeAlias::x`
pub use std::vec::Vec;
