//@ aux-build:reference_label_typo_external_macro.rs
#![deny(rustdoc::broken_intra_doc_links)]

extern crate reference_label_typo_external_macro as dep;

#[doc = dep::reference_docs!(
    "[configration].\n\n[configuration]: https://example.com/configuration"
)]
//~^^ ERROR unresolved link to `configration`
//~| HELP reference label `configuration`
pub struct FromCallerArgument;
