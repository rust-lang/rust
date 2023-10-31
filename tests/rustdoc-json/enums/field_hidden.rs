// Regression test for <https://github.com/rust-lang/rust/issues/100529>.

#![no_core]
#![feature(no_core, lang_items)]

#[lang = "sized"]
trait Sized {}

// @has "$.index[*][?(@.name=='ParseError')]"
// @has "$.index[*][?(@.name=='UnexpectedEndTag')]"
// @is "$.index[*][?(@.name=='UnexpectedEndTag')].inner.variant.kind.tuple" [null]
// @is "$.index[*][?(@.name=='UnexpectedEndTag')].inner.variant.discriminant" null

pub enum ParseError {
    UnexpectedEndTag(#[doc(hidden)] u32),
}
