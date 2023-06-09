// This is a regression test for <https://github.com/rust-lang/rust/issues/106373>.
// It ensures that the items in the `doc(hidden)` const block don't show up in the
// generated docs.

// compile-flags: --document-private-items

#![crate_name = "foo"]

// @has 'foo/index.html'
// @count - '//*[@class="item-table"]//a[@class="struct"]' 2
// @count - '//*[@class="item-table"]//a[@class="trait"]' 1
// @count - '//*[@class="item-table"]//a[@class="macro"]' 0
#[doc(hidden)]
const _: () = {
    macro_rules! stry {
        () => {};
    }

    struct ShouldBeHidden;

    // @has 'foo/struct.Foo.html'
    // @!has - '//*[@class="code-header"]' 'impl Bar for Foo'
    #[doc(hidden)]
    impl Bar for Foo {
        fn bar(&self) {
            struct SHouldAlsoBeHidden;
        }
    }

    // @has 'foo/struct.Private.html'
    // @has - '//*[@id="impl-Bar-for-Private"]/*[@class="code-header"]' 'impl Bar for Private'
    // @has - '//*[@id="method.bar"]/*[@class="code-header"]' 'fn bar(&self)'
    impl Bar for Private {
        fn bar(&self) {}
    }

    // @has - '//*[@id="impl-Private"]/*[@class="code-header"]' 'impl Private'
    // @has - '//*[@id="method.tralala"]/*[@class="code-header"]' 'fn tralala()'
    impl Private {
        fn tralala() {}
    }
};


struct Private;
pub struct Foo;

pub trait Bar {
    fn bar(&self);
}
