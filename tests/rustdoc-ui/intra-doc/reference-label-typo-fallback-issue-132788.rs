//@ forbid-output: [configuration].
#![deny(rustdoc::broken_intra_doc_links)]

#[doc = concat!(
    "[config",
    "ration].\n\n",
    "[configuration]: https://example.com/configuration"
)]
//~^^^^^ ERROR unresolved link to `configration`
//~| HELP reference label `configuration`
pub struct Generated;
