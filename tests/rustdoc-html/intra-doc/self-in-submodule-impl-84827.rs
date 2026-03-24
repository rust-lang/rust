// https://github.com/rust-lang/rust/issues/84827
// `Self` in a doc link inside an impl block in a submodule should resolve
// correctly even when the type is not in scope in that submodule.

#![crate_name = "foo"]
#![deny(rustdoc::broken_intra_doc_links)]

pub struct Foo {
    pub foo: i32,
}

pub mod bar {
    impl crate::Foo {
        //@ has foo/struct.Foo.html '//a[@href="struct.Foo.html#structfield.foo"]' 'Self::foo'
        /// Baz the [`Self::foo`] field.
        pub fn baz(&self) {
            let _ = self.foo;
        }

        //@ has foo/struct.Foo.html '//a[@href="struct.Foo.html#method.baz"]' 'Self::baz'
        /// See also [`Self::baz`].
        pub fn qux(&self) {}
    }
}
