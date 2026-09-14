#![feature(rustc_attrs, const_trait_impl)]
#![allow(internal_features)]
#![rustc_dump_variances_of_opaques]

const trait Foo {}

const impl Foo for () {}

fn foo<'a: 'a>() -> impl const Foo {}
//~^ ERROR ['a: *]

fn main() {}
