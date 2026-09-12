#![deny(rustdoc::broken_intra_doc_links)]

/// Hello [arg@x]
//~^ ERROR `arg@` disambiguators are experimental
pub fn a(x: ()) {}
