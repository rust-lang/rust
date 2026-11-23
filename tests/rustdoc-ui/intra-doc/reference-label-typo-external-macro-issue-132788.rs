//@ aux-build:reference_label_typo_external_macro.rs
//@ forbid-output: [configuration].
#![deny(rustdoc::broken_intra_doc_links)]

extern crate reference_label_typo_external_macro;

#[doc = reference_label_typo_external_macro::reference_docs!()]
//~? ERROR unresolved link to `configration`
//~? HELP reference label `configuration`
pub struct FromExternalMacro;
