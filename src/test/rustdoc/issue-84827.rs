#![deny(broken_intra_doc_links)]

// @has issue_84827/struct.Foo.html
// @has - '//a[@href="struct.Foo.html#structfield.foo"]' 'Self::foo'

pub struct Foo {
    pub foo: i32,
}

pub mod bar {
    impl crate::Foo {
        /// [`Self::foo`].
        pub fn baz(&self) {}
    }
}
