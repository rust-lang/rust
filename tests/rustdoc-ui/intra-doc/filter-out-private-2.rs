// This test ensures that ambiguities (not) resolved at a later stage still emit an error.

#![deny(rustdoc::broken_intra_doc_links)]
#![crate_name = "foo"]

#[doc(hidden)]
pub struct Thing {}

#[allow(non_snake_case)]
#[doc(hidden)]
pub fn Thing() {}

/// Do stuff with [`Thing`].
//~^ ERROR all items matching `Thing` are private or doc(hidden)
pub fn repro(_: Thing) {}
