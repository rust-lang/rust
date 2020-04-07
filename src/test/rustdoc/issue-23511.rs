#![feature(lang_items)]
#![no_std]

pub mod str {
    #![doc(primitive = "str")]

    #[lang = "str_alloc_impl"]
    impl str {
        // @has search-index.js foo
        pub fn foo(&self) {}
    }
}
