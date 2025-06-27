// Regression test for <https://github.com/rust-lang/rust/issues/100529>.

//@ jq .index[] | select(.name == "ParseError")
//@ arg unexpected_end_tag .index[] | select(.name == "UnexpectedEndTag")
//@ jq $unexpected_end_tag
//@ jq $unexpected_end_tag.inner.variant.kind?.tuple == [null]
//@ jq $unexpected_end_tag.inner.variant.discriminant? == null

pub enum ParseError {
    UnexpectedEndTag(#[doc(hidden)] u32),
}
