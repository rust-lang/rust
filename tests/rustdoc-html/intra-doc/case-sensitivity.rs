// This test ensures that links are case sensitive and that intra-doc links don't get overriden
// by a link definition below which the same when case insensitive.
// Regression test for <https://github.com/rust-lang/rust/issues/80882>.

#![crate_name = "foo"]

//@ has 'foo/index.html'
//@ has - '//*[@class="item-table"]/dd/a[@href="trait.Foo.html"]' 'Foo'
//@ has - '//*[@class="item-table"]/dd/a[@href="trait.Bar.html#tymethod.foo"]' 'foo'

/// [`Foo`] and [`foo`]
///
/// [`foo`]: Bar::foo
pub trait Bar {
    fn foo();
}

// We want to ensure that the link is correctly resolved in case the link def doesn't
// case-sensitively match.
//@ has - '//*[@class="item-table"]/dd/a[@href="https://cookie.land"]' 'Flower'

/// [`Flower`]
///
/// [`flower`]: https://cookie.land
pub trait Foo {}
