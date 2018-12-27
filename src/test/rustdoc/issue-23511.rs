#![feature(lang_items)]
#![no_std]

pub mod str {
    #![doc(primitive = "str")]

    #[lang = "str_alloc"]
    impl str {
        // @has search-index.js foo
        pub fn foo(&self) {}
    }
}
