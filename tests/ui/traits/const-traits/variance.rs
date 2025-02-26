#![feature(rustc_attrs, const_trait_impl)]
#![allow(internal_features)]
#![rustc_variance_of_opaques]

#[const_trait]
trait Foo {}

impl const Foo for () {}

fn foo<'a: 'a>() -> impl const Foo {}
//~^ ERROR ['a: *]

fn main() {}
