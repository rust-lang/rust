//@ compile-flags: -Zunpretty=hir
//@ check-pass

// FIXME(jdonszelmann): the pretty printing output for deprecated (and possibly more attrs) is
// slightly broken.
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
