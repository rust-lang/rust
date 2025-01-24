//@ edition: 2024

#![feature(rustc_attrs)]
#![feature(type_alias_impl_trait)]
#![rustc_variance_of_opaques]

fn foo(x: &()) -> impl IntoIterator<Item = impl Sized> + use<> {
    //~^ ERROR ['_: o]
    //~| ERROR ['_: o]
    //~| ERROR `impl Trait` captures lifetime parameter
    [*x]
}

fn main() {}
