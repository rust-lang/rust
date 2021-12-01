// check-pass
#![deny(rustdoc::broken_intra_doc_links)]

mod bar {
    macro_rules! str {() => {}}
}

pub mod foo {
    pub struct Baz {}
    impl Baz {
        /// [str]
        pub fn foo(&self) {}
    }
}
