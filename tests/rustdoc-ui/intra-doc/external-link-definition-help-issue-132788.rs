// Regression test for https://github.com/rust-lang/rust/issues/132788.
#![deny(rustdoc::broken_intra_doc_links)]

// An unresolved name should explain the external-link alternative.
/// Compare with [ForeignWidget].
//~^ ERROR unresolved link to `ForeignWidget`
//~| HELP to escape `[` and `]` characters
//~| HELP if you intended to link to external documentation
pub struct MissingDefinition;

// Providing an explicit destination should resolve the Markdown link.
/// Compare with [ForeignWidget].
///
/// [ForeignWidget]: https://example.com/foreign-widget
pub struct WithDefinition;

// Adding a reference definition would not fix an inline destination.
/// Compare with [the foreign widget](ForeignWidget).
//~^ ERROR unresolved link to `ForeignWidget`
//~| HELP to escape `[` and `]` characters
pub struct InlineDestination;
