#![crate_name = "foo"]

mod hidden {
    // @has foo/hidden/struct.Foo.html
    // @has - '//p/a' '../../foo/struct.FooBar.html'
    pub struct Foo {}

    // @has foo/hidden/bar/index.html
    // @has - '//p/a' '../../foo/baz/index.html'
    pub mod bar {
        // @has foo/hidden/bar/struct.Thing.html
        // @has - '//p/a' '../../foo/baz/struct.Thing.html'
        pub struct Thing {}
    }
}

// @has foo/struct.FooBar.html
pub use hidden::Foo as FooBar;

// @has foo/baz/index.html
// @has foo/baz/struct.Thing.html
pub use hidden::bar as baz;
