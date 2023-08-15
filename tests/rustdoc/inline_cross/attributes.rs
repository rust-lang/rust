// aux-crate:attributes=attributes.rs
// edition:2021
#![crate_name = "user"]

// @has 'user/struct.NonExhaustive.html'
// @has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[non_exhaustive]'
pub use attributes::NonExhaustive;
