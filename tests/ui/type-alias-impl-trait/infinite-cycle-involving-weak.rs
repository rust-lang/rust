#![feature(type_alias_impl_trait)]

type T = impl Copy;
//~^ ERROR cannot resolve opaque type

#[defines(T)]
static STATIC: T = None::<&'static T>;

fn main() {}
