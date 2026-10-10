pub mod foo {
    pub struct Foo;
}

//@ has please_inline/a/index.html
pub mod a {
    //@ !hasraw - 'pub use foo::'
    //@ has please_inline/a/type.Foo.html
    #[doc(inline)]
    pub use foo::Foo;
}

//@ has please_inline/b/index.html
pub mod b {
    //@ hasraw - 'pub use foo::'
    //@ !has please_inline/b/type.Foo.html
    #[feature(inline)]
    pub use foo::Foo;
}
