#![feature(type_alias_impl_trait)]
//~^ ERROR overflow normalizing the opaque type

type T = impl Copy;
//~^ ERROR cannot resolve opaque type

static STATIC: T = None::<&'static T>;

fn main() {}
