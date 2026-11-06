#![deny(rustdoc::broken_intra_doc_links)]

/// [configration].
//~^ ERROR unresolved link to `configration`
//~| HELP reference label `configuration`
///
/// [configration][].
//~^ ERROR unresolved link to `configration`
//~| HELP reference label `configuration`
///
/// [Read][configration].
//~^ ERROR unresolved link to `configration`
//~| HELP reference label `configuration`
///
/// [configuration].
///
/// [configuration]: https://example.com/configuration
pub struct Guide;
