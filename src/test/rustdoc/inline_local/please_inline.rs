pub mod foo {
    pub struct Foo;
}

// @has please_inline/a/index.html
pub mod a {
    // @!has - 'pub use foo::'
    // @has please_inline/a/struct.Foo.html
    #[doc(inline)]
    pub use foo::Foo;
}

// @has please_inline/b/index.html
pub mod b {
    // @has - 'pub use foo::'
    // @!has please_inline/b/struct.Foo.html
    #[feature(inline)]
    pub use foo::Foo;
}
