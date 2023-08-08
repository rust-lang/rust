#![feature(rustc_attrs)]
#![feature(rustdoc_internals)]
#![no_std]

pub mod str {
    #![rustc_doc_primitive = "str"]

    impl str {
        // @hasraw search-index.js foo
        #[rustc_allow_incoherent_impl]
        pub fn foo(&self) {}
    }
}
