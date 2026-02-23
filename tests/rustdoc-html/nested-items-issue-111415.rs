// Regression test for <https://github.com/rust-lang/rust/issues/111415>.
// This test ensures that only impl blocks are documented in bodies.

#![crate_name = "foo"]

//@ has 'foo/index.html'
// Checking there are only three sections.
//@ count - '//*[@id="main-content"]/*[@class="section-header"]' 3
//@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Structs'
//@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Functions'
//@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Traits'
// Checking that there are only three items.
//@ count - '//*[@id="main-content"]//dt' 3
//@ has - '//*[@id="main-content"]//a[@href="struct.Bar.html"]' 'Bar'
//@ has - '//*[@id="main-content"]//a[@href="fn.foo.html"]' 'foo'
//@ has - '//*[@id="main-content"]//a[@href="trait.Foo.html"]' 'Foo'

// Now checking that the `foo` method is visible in `Bar` page.
//@ has 'foo/struct.Bar.html'
//@ has - '//*[@id="method.foo"]/*[@class="code-header"]' 'pub fn foo()'
//@ has - '//*[@id="method.bar"]/*[@class="code-header"]' 'fn bar()'
pub struct Bar;

pub trait Foo {
    fn bar() {}
}

pub fn foo() {
    pub mod inaccessible {}
    pub fn inner() {}
    pub const BAR: u32 = 0;
    impl Bar {
        pub fn foo() {}
    }
    impl Foo for Bar {}
}
