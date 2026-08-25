// Regression test for <https://github.com/rust-lang/rust/issues/80882>.

//@ check-pass

#![deny(rustdoc::broken_intra_doc_links)]

// We want to ensure that there is no warning emitted in case we first check that
// `Flower` is not an intra-doc link.

/// [`Flower`]
///
/// [`flower`]: https://cookie.land
pub trait Foo {}
