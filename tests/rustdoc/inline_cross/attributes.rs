// Ensure that we render attributes on inlined cross-crate re-exported items.
// issue: <https://github.com/rust-lang/rust/issues/144004>

//@ aux-crate:attributes=attributes.rs
//@ edition:2021
#![crate_name = "user"]

//@ has 'user/fn.no_mangle.html' '//pre[@class="rust item-decl"]' '#[unsafe(no_mangle)]'
pub use attributes::no_mangle;

//@ has 'user/fn.link_section.html' '//pre[@class="rust item-decl"]' \
//                                  '#[unsafe(link_section = ".here")]'
pub use attributes::link_section;

//@ has 'user/fn.export_name.html' '//pre[@class="rust item-decl"]' \
//                                 '#[unsafe(export_name = "exonym")]'
pub use attributes::export_name;

//@ has 'user/struct.NonExhaustive.html' '//pre[@class="rust item-decl"]' '#[non_exhaustive]'
pub use attributes::NonExhaustive;
