//@ check-pass
// https://github.com/rust-lang/rust/issues/84827

// rustdoc should resolve `Self` in doc links inside an `impl` block that lives
// in a submodule where the type is not directly in scope.

#![deny(rustdoc::broken_intra_doc_links)]

pub struct Foo {
    pub foo: i32,
}

pub mod bar {
    impl crate::Foo {
        /// Baz the [`Self::foo`] field.
        pub fn baz(&self) {
            let _ = self.foo;
        }

        /// Returns a new [`Self`].
        pub fn new() -> Self {
            Self { foo: 0 }
        }

        /// See also [`Self::baz`].
        pub fn qux(&self) {}
    }
}
