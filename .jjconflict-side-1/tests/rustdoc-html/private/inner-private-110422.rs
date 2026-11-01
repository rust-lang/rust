// Regression test for <https://github.com/rust-lang/rust/issues/110422>.
// This test ensures that inner items (except for implementations and macros)
// don't appear in documentation.

//@ compile-flags: --document-private-items

#![crate_name = "foo"]

//@ has 'foo/index.html'
// Checking there is no "trait" entry.
//@ count - '//*[@id="main-content"]/*[@class="section-header"]' 4
//@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Structs'
//@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Constants'
//@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Functions'
//@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Macros'

//@ has - '//a[@href="fn.foo.html"]' 'foo'
fn foo() {
    fn bar() {}

    //@ has - '//a[@class="macro"]' 'visible_macro'
    //@ !has - '//a[@class="macro"]' 'non_visible_macro'
    //@ has 'foo/macro.visible_macro.html'
    //@ !has 'foo/macro.non_visible_macro.html'
    #[macro_export]
    macro_rules! visible_macro {
        () => {}
    }

    macro_rules! non_visible_macro {
        () => {}
    }
}

//@ has 'foo/index.html'
//@ has - '//a[@href="struct.Bar.html"]' 'Bar'
struct Bar;

const BAR: i32 = {
    //@ !has - '//a[@href="fn.yo.html"]' 'yo'
    //@ !has 'foo/fn.yo.html'
    fn yo() {}

    //@ !has 'foo/index.html' '//a[@href="trait.Foo.html"]' 'Foo'
    //@ !has 'foo/trait.Foo.html'
    trait Foo {
        fn babar() {}
    }
    impl Foo for Bar {}

    //@ has 'foo/struct.Bar.html'
    //@ has - '//*[@id="method.foo"]/*[@class="code-header"]' 'pub(crate) fn foo()'
    //@ count - '//*[@id="main-content"]/*[@class="section-header"]' 3
    // We now check that the `Foo` trait is not documented nor visible on `Bar` page.
    //@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Implementations'
    //@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Auto Trait Implementations'
    //@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Blanket Implementations'
    //@ !has - '//*[@href="trait.Foo.html#method.babar"]/*[@class="code-header"]' 'fn babar()'
    impl Bar {
        fn foo() {}
    }

    1
};
