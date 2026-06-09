// This test ensures that the intra-doc link in `deprecated` note is resolved at the correct
// location (ie in the current crate and not in the reexported item's location/crate) and
// therefore doesn't crash.
//
// This is a regression test for <https://github.com/rust-lang/rust/issues/151028>.

#![crate_name = "foo"]

#[deprecated(note = "use [`std::mem::forget`]")]
#[doc(inline)]
pub use std::mem::drop;
