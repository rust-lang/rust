//@ edition: 2021
//@ aux-crate: edition_redirect_chain=edition-redirect-chain.rs

#![crate_name = "foo"]

// The selected redirect must preserve its own import chain across crate boundaries,
// including the redirect import itself and intermediate reexports of its target.
//@ has 'foo/struct.Item.html' '//div[@class="docblock"]' \
//  'Downstream documentation. Redirect documentation. Intermediate documentation. \
//  Target documentation.'
//@ !has 'foo/struct.Item.html' '//div[@class="docblock"]' 'Default import documentation.'
//@ !has 'foo/struct.Item.html' '//div[@class="docblock"]' 'Default target documentation.'
/// Downstream documentation.
#[doc(inline)]
pub use edition_redirect_chain::Item;
