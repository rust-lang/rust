//@ forbid-output: similarly named reference label
#![deny(rustdoc::broken_intra_doc_links)]

/// [Read](configration).
//~^ ERROR unresolved link to `configration`
///
/// [configuration]: https://example.com/configuration
pub struct InlineDestination;

/// [zebra].
//~^ ERROR unresolved link to `zebra`
///
/// [configuration]: https://example.com/configuration
pub struct NoCloseLabel;

/// [configration].
//~^ ERROR unresolved link to `configration`
pub struct SeparateDocumentation;

pub struct Widget;

/// [Widget].
///
/// [Widgets]: https://example.com/widgets
pub struct RustItemLink;
