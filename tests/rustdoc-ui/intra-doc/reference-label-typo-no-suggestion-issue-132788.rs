//@ forbid-output: similarly named reference label
#![deny(rustdoc::broken_intra_doc_links)]

/// [Read](configration).
//~^ ERROR unresolved link to `configration`
//~| HELP to escape `[` and `]` characters
///
/// [configuration]: https://example.com/configuration
pub struct InlineDestination;

/// [zebra].
//~^ ERROR unresolved link to `zebra`
//~| HELP to escape `[` and `]` characters
//~| HELP if you intended to link to external documentation
///
/// [configuration]: https://example.com/configuration
pub struct NoCloseLabel;

/// [configration].
//~^ ERROR unresolved link to `configration`
//~| HELP to escape `[` and `]` characters
//~| HELP if you intended to link to external documentation
pub struct SeparateDocumentation;

pub struct Widget;

/// [Widget].
///
/// [Widgets]: https://example.com/widgets
pub struct RustItemLink;
