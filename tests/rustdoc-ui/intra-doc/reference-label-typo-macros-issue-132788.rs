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

macro_rules! reference_docs {
    () => {
        "[configration].\n\n[configuration]: https://example.com/configuration"
        //~^ ERROR unresolved link to `configration`
        //~| HELP reference label `configuration`
    };
}

#[doc = reference_docs!()]
pub struct FromMacroValue;

macro_rules! document_argument {
    ($docs:literal) => {
        #[doc = $docs]
        pub struct FromArgument;
    };
}

document_argument!("[configration].\n\n[configuration]: https://example.com/configuration");
//~^ ERROR unresolved link to `configration`
//~| HELP reference label `configuration`
