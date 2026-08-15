//@ count "$.index[?(@.name=='generic_returns')].inner.module.items[*]" 2

//@ set foo = "$.index[?(@.name=='Foo')].id"
pub trait Foo {}

//@ is "$.index[?(@.name=='get_foo')].inner.function.sig.inputs" []
//@ count "$.index[?(@.name=='get_foo')].inner.function.sig.output.impl_trait[*]" 1
//@ is "$.index[?(@.name=='get_foo')].inner.function.sig.output.impl_trait[0].trait_bound.trait.id" $foo
pub fn get_foo() -> impl Foo {
    Fooer {}
}

struct Fooer {}

impl Foo for Fooer {}
