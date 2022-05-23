#![feature(rustc_attrs)]
#![feature(rustdoc_internals)]
#![no_std]

pub mod str {
    #![doc(primitive = "str")]

    impl str {
        // @has search-index.js foo
        #[rustc_allow_incoherent_impl]
        pub fn foo(&self) {}
    }
}
