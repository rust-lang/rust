// Regression test for <https://github.com/rust-lang/rust/issues/100529>.

#![no_core]
#![feature(no_core)]

// @has "$.index[*][?(@.name=='ParseError')]"
// @has "$.index[*][?(@.name=='UnexpectedEndTag')]"
// @is "$.index[*][?(@.name=='UnexpectedEndTag')].inner.kind" '"tuple"'
// @is "$.index[*][?(@.name=='UnexpectedEndTag')].inner.fields" []
// @is "$.index[*][?(@.name=='UnexpectedEndTag')].inner.fields_stripped" true

pub enum ParseError {
    UnexpectedEndTag(#[doc(hidden)] u32),
}
