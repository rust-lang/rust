// https://github.com/rust-lang/rust/issues/132788
#![deny(rustdoc::broken_intra_doc_links)]

macro_rules! generated_doc {
    () => {
        #[doc = "Access arr[idx]."]
        //~^ ERROR unresolved link to `idx`
        //~| HELP try marking as source code
        pub struct Generated;
    };
}

generated_doc!();
