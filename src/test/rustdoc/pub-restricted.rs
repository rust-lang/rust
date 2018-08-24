// ignore-tidy-linelength

// compile-flags: --document-private-items

#![feature(crate_visibility_modifier)]

#![crate_name = "foo"]

// @has 'foo/struct.FooPublic.html' '//pre' 'pub struct FooPublic'
pub struct FooPublic;
// @has 'foo/struct.FooJustCrate.html' '//pre' 'pub(crate) struct FooJustCrate'
crate struct FooJustCrate;
// @has 'foo/struct.FooPubCrate.html' '//pre' 'pub(crate) struct FooPubCrate'
pub(crate) struct FooPubCrate;
// @has 'foo/struct.FooSelf.html' '//pre' 'pub(self) struct FooSelf'
pub(self) struct FooSelf;
// @has 'foo/struct.FooInSelf.html' '//pre' 'pub(self) struct FooInSelf'
pub(in self) struct FooInSelf;
mod a {
    // @has 'foo/a/struct.FooSuper.html' '//pre' 'pub(super) struct FooSuper'
    pub(super) struct FooSuper;
    // @has 'foo/a/struct.FooInSuper.html' '//pre' 'pub(super) struct FooInSuper'
    pub(in super) struct FooInSuper;
    // @has 'foo/a/struct.FooInA.html' '//pre' 'pub(in a) struct FooInA'
    pub(in a) struct FooInA;
    mod b {
        // @has 'foo/a/b/struct.FooInSelfSuperB.html' '//pre' 'pub(in self::super::b) struct FooInSelfSuperB'
        pub(in self::super::b) struct FooInSelfSuperB;
        // @has 'foo/a/b/struct.FooInSuperSuper.html' '//pre' 'pub(in super::super) struct FooInSuperSuper'
        pub(in super::super) struct FooInSuperSuper;
        // @has 'foo/a/b/struct.FooInAB.html' '//pre' 'pub(in a::b) struct FooInAB'
        pub(in a::b) struct FooInAB;
    }
}
