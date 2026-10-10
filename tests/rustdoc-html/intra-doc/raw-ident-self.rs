#![deny(rustdoc::broken_intra_doc_links)]
pub mod r#impl {
    pub struct S;

    impl S {
        /// See [Self::b].
        //@ has raw_ident_self/impl/type.S.html
        //@ has - '//a[@href="type.S.html#method.b"]' 'Self::b'
        pub fn a() {}

        pub fn b() {}
    }
}
