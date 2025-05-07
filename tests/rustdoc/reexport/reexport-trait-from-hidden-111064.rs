// Regression test for <https://github.com/rust-lang/rust/issues/111064>.
// Methods from a re-exported trait inside a `#[doc(hidden)]` item should
// be visible.

#![crate_name = "foo"]

//@ has 'foo/index.html'
//@ has - '//*[@id="main-content"]//dt/a[@href="trait.Foo.html"]' 'Foo'

//@ has 'foo/trait.Foo.html'
//@ has - '//*[@id="main-content"]//*[@class="code-header"]' 'fn test()'

#[doc(hidden)]
mod hidden {
    pub trait Foo {
        /// Hello, world!
        fn test();
    }
}

pub use hidden::Foo;
