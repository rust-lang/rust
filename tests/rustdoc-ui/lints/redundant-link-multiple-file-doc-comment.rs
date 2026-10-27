// This test ensures that if a doc comment is split between inner and outer attributes,
// it still works, even if the attributes are in different files.
// Regression test for <https://github.com/rust-lang/rust/issues/164282>.

#![deny(rustdoc::redundant_explicit_links)]

/// Outer docs on the module.
#[path = "auxiliary/redundant-link-multiple-file-doc-comment.rs"]
pub mod m;
//~? ERROR: redundant_explicit_links
pub use m::Foo;

/// Outer docs on the module.
pub mod a {
    //! See [`A`](crate::a::A).
    //~^ ERROR: redundant_explicit_links
    pub struct A;
}
pub use a::A;
