//@ edition: 2024
//@ compile-flags: -Zunstable-options

#![feature(rustc_attrs)]
#![feature(type_alias_impl_trait)]
#![rustc_variance_of_opaques]

fn foo(x: &()) -> impl IntoIterator<Item = impl Sized> + use<> {
    //~^ ERROR []
    //~| ERROR []
    [*x]
}

fn main() {}
