#![deny(rustdoc::broken_intra_doc_links)]
pub mod r#impl {
    pub struct S;

    impl S {
        /// See [Self::b].
        // @has raw_ident_self/impl/struct.S.html
        // @has - '//a[@href="struct.S.html#method.b"]' 'Self::b'
        pub fn a() {}

        pub fn b() {}
    }
}
