// https://github.com/rust-lang/rust/issues/84827
//
// Regression test for resolving `Self` in intra-doc links inside an `impl`
// block in a submodule. The `Self` type comes from the impl, not from names
// in scope, so:
//   1. It resolves even when the implemented type is not in scope.
//   2. It resolves to the actual `Self` type, not to a different type with
//      the same name that happens to be in scope.

#![crate_name = "foo"]
#![deny(rustdoc::broken_intra_doc_links)]
#![allow(unused_imports)]

pub struct Foo {
    pub foo: i32,
}

// Case 1: `Foo` is not in scope inside `bar`, but `Self::foo` still resolves
// to the field on `crate::Foo`.
pub mod bar {
    //@ has foo/struct.Foo.html '//a[@href="struct.Foo.html#structfield.foo"]' 'Self::foo'
    impl crate::Foo {
        /// Baz the [`Self::foo`].
        pub fn baz(&self) {}
    }
}

// Case 2: A different type named `Foo` is in scope inside `baz`. `Self::foo`
// must still resolve to `crate::Foo::foo`, not to `crate::other::Foo` (which
// is a unit struct and has no `foo` field).
pub mod baz {
    use crate::other::Foo;

    //@ has foo/struct.Foo.html '//a[@href="struct.Foo.html#structfield.foo"]' 'Self::foo'
    impl crate::Foo {
        /// Quux the [`Self::foo`].
        pub fn quux(&self) {}
    }
}

pub mod other {
    pub struct Foo;
}
