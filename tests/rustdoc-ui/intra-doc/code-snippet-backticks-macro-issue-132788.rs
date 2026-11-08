//@ forbid-output: try marking as source code
// https://github.com/rust-lang/rust/issues/132788
#![deny(rustdoc::broken_intra_doc_links)]

macro_rules! generated_doc {
    () => {
        #[doc = "Access arr[idx]."]
        //~^ ERROR unresolved link to `idx`
        //~| HELP to escape `[` and `]` characters
        pub struct Generated;
    };
}

generated_doc!();
