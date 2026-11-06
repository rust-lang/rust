//@ forbid-output: [configuration].
#![deny(rustdoc::broken_intra_doc_links)]

macro_rules! documented_item {
    () => {
        /// [configration].
        //~^ ERROR unresolved link to `configration`
        //~| HELP reference label `configuration`
        ///
        /// [configuration]: https://example.com/configuration
        pub struct Generated;
    };
}

documented_item!();
