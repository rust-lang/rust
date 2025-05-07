//@ compile-flags: -Zunpretty=hir
//@ check-pass
//@ edition: 2015

#[deprecated]
pub struct PlainDeprecated;

#[deprecated = "here's why this is deprecated"]
pub struct DirectNote;

#[deprecated(note = "here's why this is deprecated")]
pub struct ExplicitNote;

#[deprecated(since = "1.2.3", note = "here's why this is deprecated")]
pub struct SinceAndNote;

#[deprecated(note = "here's why this is deprecated", since = "1.2.3")]
pub struct FlippedOrder;
