// Test that we do not currently display `~const` in rustdoc
// as that syntax is currently provisional; `~const Destruct` has
// no effect on stable code so it should be hidden as well.
//
// To future blessers: make sure that `const_trait_impl` is
// stabilized when changing `@!has` to `@has`, and please do
// not remove this test.
//
// FIXME(const_trait_impl) add `const_trait` to `Fn` so we use `~const`
// FIXME(const_trait_impl) restore `const_trait` to `Destruct`
#![feature(const_trait_impl)]
#![crate_name = "foo"]

use std::marker::Destruct;

pub struct S<T>(T);

//@ !has foo/trait.Tr.html '//pre[@class="rust item-decl"]/code/a[@class="trait"]' '~const'
//@ has - '//pre[@class="rust item-decl"]/code/a[@class="trait"]' 'Fn'
//@ !has - '//pre[@class="rust item-decl"]/code/span[@class="where"]' '~const'
//@ has - '//pre[@class="rust item-decl"]/code/span[@class="where"]' ': Fn'
#[const_trait]
pub trait Tr<T> {
    //@ !has - '//section[@id="method.a"]/h4[@class="code-header"]' '~const'
    //@ has - '//section[@id="method.a"]/h4[@class="code-header"]/a[@class="trait"]' 'Fn'
    //@ !has - '//section[@id="method.a"]/h4[@class="code-header"]/span[@class="where"]' '~const'
    //@ has - '//section[@id="method.a"]/h4[@class="code-header"]/div[@class="where"]' ': Fn'
    fn a<A: /* ~const */ Fn() /* + ~const Destruct */>()
    where
        Option<A>: /* ~const */ Fn() /* + ~const Destruct */,
    {
    }
}

//@ has - '//section[@id="impl-Tr%3CT%3E-for-T"]' ''
//@ !has - '//section[@id="impl-Tr%3CT%3E-for-T"]/h3[@class="code-header"]' '~const'
//@ has - '//section[@id="impl-Tr%3CT%3E-for-T"]/h3[@class="code-header"]/a[@class="trait"]' 'Fn'
//@ !has - '//section[@id="impl-Tr%3CT%3E-for-T"]/h3[@class="code-header"]/span[@class="where"]' '~const'
//@ has - '//section[@id="impl-Tr%3CT%3E-for-T"]/h3[@class="code-header"]/div[@class="where"]' ': Fn'
impl<T: /* ~const */ Fn() /* + ~const Destruct */> const Tr<T> for T
where
    Option<T>: /* ~const */ Fn() /* + ~const Destruct */,
{
    fn a<A: /* ~const */ Fn() /* + ~const Destruct */>()
    where
        Option<A>: /* ~const */ Fn() /* + ~const Destruct */,
    {
    }
}

//@ !has foo/fn.foo.html '//pre[@class="rust item-decl"]/code/a[@class="trait"]' '~const'
//@ has - '//pre[@class="rust item-decl"]/code/a[@class="trait"]' 'Fn'
//@ !has - '//pre[@class="rust item-decl"]/code/div[@class="where"]' '~const'
//@ has - '//pre[@class="rust item-decl"]/code/div[@class="where"]' ': Fn'
pub const fn foo<F: /* ~const */ Fn() /* + ~const Destruct */>()
where
    Option<F>: /* ~const */ Fn() /* + ~const Destruct */,
{
    F::a()
}

impl<T> S<T> {
    //@ !has foo/struct.S.html '//section[@id="method.foo"]/h4[@class="code-header"]' '~const'
    //@ has - '//section[@id="method.foo"]/h4[@class="code-header"]/a[@class="trait"]' 'Fn'
    //@ !has - '//section[@id="method.foo"]/h4[@class="code-header"]/span[@class="where"]' '~const'
    //@ has - '//section[@id="method.foo"]/h4[@class="code-header"]/div[@class="where"]' ': Fn'
    pub const fn foo<B, C: /* ~const */ Fn() /* + ~const Destruct */>()
    where
        B: /* ~const */ Fn() /* + ~const Destruct */,
    {
        B::a()
    }
}
