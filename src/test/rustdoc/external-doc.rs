#![feature(external_doc)]

// @has external_doc/struct.CanHasDocs.html
// @has - '//h1' 'External Docs'
// @has - '//h2' 'Inline Docs'
#[doc(include = "auxiliary/external-doc.md")]
/// ## Inline Docs
pub struct CanHasDocs;
