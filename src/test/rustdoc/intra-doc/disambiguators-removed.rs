// ignore-tidy-linelength
#![deny(intra_doc_link_resolution_failure)]
// first try backticks
/// Trait: [`trait@Name`], fn: [`fn@Name`], [`Name`][`macro@Name`]
// @has disambiguators_removed/struct.AtDisambiguator.html
// @has - '//a[@href="../disambiguators_removed/trait.Name.html"][code]' "Name"
// @has - '//a[@href="../disambiguators_removed/fn.Name.html"][code]' "Name"
// @has - '//a[@href="../disambiguators_removed/macro.Name.html"][code]' "Name"
pub struct AtDisambiguator;

/// fn: [`Name()`], macro: [`Name!`]
// @has disambiguators_removed/struct.SymbolDisambiguator.html
// @has - '//a[@href="../disambiguators_removed/fn.Name.html"][code]' "Name()"
// @has - '//a[@href="../disambiguators_removed/macro.Name.html"][code]' "Name!"
pub struct SymbolDisambiguator;

// Now make sure that backticks aren't added if they weren't already there
/// [fn@Name]
// @has disambiguators_removed/trait.Name.html
// @has - '//a[@href="../disambiguators_removed/fn.Name.html"]' "Name"
// @!has - '//a[@href="../disambiguators_removed/fn.Name.html"][code]' "Name"

// FIXME: this will turn !() into ! alone
/// [Name!()]
// @has - '//a[@href="../disambiguators_removed/macro.Name.html"]' "Name!"
pub trait Name {}

#[allow(non_snake_case)]

// Try collapsed reference links
/// [macro@Name][]
// @has disambiguators_removed/fn.Name.html
// @has - '//a[@href="../disambiguators_removed/macro.Name.html"]' "Name"

// Try links that have the same text as a generated URL
/// Weird URL aligned [../disambiguators_removed/macro.Name.html][trait@Name]
// @has - '//a[@href="../disambiguators_removed/trait.Name.html"]' "../disambiguators_removed/macro.Name.html"
pub fn Name() {}

#[macro_export]
// Rustdoc doesn't currently handle links that have weird interspersing of inline code blocks.
/// [fn@Na`m`e]
// @has disambiguators_removed/macro.Name.html
// @has - '//a[@href="../disambiguators_removed/fn.Name.html"]' "fn@Name"

// It also doesn't handle any case where the code block isn't the whole link text:
/// [trait@`Name`]
// @has - '//a[@href="../disambiguators_removed/trait.Name.html"]' "trait@Name"
macro_rules! Name {
    () => ()
}
