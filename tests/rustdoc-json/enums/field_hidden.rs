// Regression test for <https://github.com/rust-lang/rust/issues/100529>.

//@ has "$.index[?(@.name=='ParseError')]"
//@ has "$.index[?(@.name=='UnexpectedEndTag')]"
//@ is "$.index[?(@.name=='UnexpectedEndTag')].inner.variant.kind.tuple" [null]
//@ is "$.index[?(@.name=='UnexpectedEndTag')].inner.variant.discriminant" null

pub enum ParseError {
    UnexpectedEndTag(#[doc(hidden)] u32),
}
