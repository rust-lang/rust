#![crate_name = "foo"]

#[doc(hidden)]
pub mod hidden {
    pub struct Foo;

    impl Foo {
        #[doc(hidden)]
        pub fn this_should_be_hidden() {}
    }

    pub struct Bar;

    impl Bar {
        fn this_should_be_hidden() {}
    }
}

//@ has foo/type.Foo.html
// Only `not_hidden` should be present.
//@ count - '//*[@id="implementations-list"]//*[@class="method"]' 1
//@ has - '//*[@id="implementations-list"]//*[@class="method"]' 'pub fn not_hidden'
//@ count - '//*[@id="rustdoc-toc"]/*[@class="block method"]//a' 1
//@ has - '//*[@id="rustdoc-toc"]/*[@class="block method"]//a' 'not_hidden'
pub use hidden::Foo;

impl Foo {
    pub fn not_hidden() {}
}

//@ has foo/type.Bar.html
//@ count - '//*[@id="implementations-list"]//*[@class="method"]' 1
//@ has - '//*[@id="implementations-list"]//*[@class="method"]' 'pub fn not_hidden'
//@ count - '//*[@id="rustdoc-toc"]/*[@class="block method"]//a' 1
//@ has - '//*[@id="rustdoc-toc"]/*[@class="block method"]//a' 'not_hidden'
pub use hidden::Bar;

impl Bar {
    pub fn not_hidden() {}
}
