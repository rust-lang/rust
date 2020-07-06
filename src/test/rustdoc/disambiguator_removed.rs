#![deny(intra_doc_link_resolution_failure)]
// first try backticks
/// Trait: [`trait@Name`], fn: [`fn@Name`], [`Name`][`macro@Name`]
// @has disambiguator_removed/struct.AtDisambiguator.html
// @has - '//a[@href="../disambiguator_removed/trait.Name.html"][code]' "Name"
// @has - '//a[@href="../disambiguator_removed/fn.Name.html"][code]' "Name"
// @has - '//a[@href="../disambiguator_removed/macro.Name.html"][code]' "Name"
pub struct AtDisambiguator;

/// fn: [`Name()`], macro: [`Name!`]
// @has disambiguator_removed/struct.SymbolDisambiguator.html
// @has - '//a[@href="../disambiguator_removed/fn.Name.html"][code]' "Name()"
// @has - '//a[@href="../disambiguator_removed/macro.Name.html"][code]' "Name!"
pub struct SymbolDisambiguator;

// Now make sure that backticks aren't added if they weren't already there
/// [fn@Name]
// @has disambiguator_removed/trait.Name.html
// @has - '//a[@href="../disambiguator_removed/fn.Name.html"]' "Name"
// @!has - '//a[@href="../disambiguator_removed/fn.Name.html"][code]' "Name"

// FIXME: this will turn !() into ! alone
/// [Name!()]
// @has - '//a[@href="../disambiguator_removed/macro.Name.html"]' "Name!"
pub trait Name {}

#[allow(non_snake_case)]
pub fn Name() {}

#[macro_export]
macro_rules! Name {
    () => ()
}
