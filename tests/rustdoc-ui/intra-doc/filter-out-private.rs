// This test ensures that ambiguities resolved at a later stage still emit an error.

#![deny(rustdoc::broken_intra_doc_links)]
#![crate_name = "foo"]

pub struct Thing {}

#[allow(non_snake_case)]
pub fn Thing() {}

/// Do stuff with [`Thing`].
//~^ ERROR `Thing` is both a function and a struct
pub fn repro(_: Thing) {}
