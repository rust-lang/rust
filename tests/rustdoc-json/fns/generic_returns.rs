//@ jq .index["\(.root)"].inner.module.items? | length == 2

//@ arg foo .index[] | select(.name == "Foo").id
pub trait Foo {}

//@ arg get_foo .index[] | select(.name == "get_foo").inner.function.sig
//@ jq $get_foo.inputs? == []
//@ jq $get_foo.output?.impl_trait[]?.trait_bound.trait?.id == $foo
pub fn get_foo() -> impl Foo {
    Fooer {}
}

struct Fooer {}

impl Foo for Fooer {}
