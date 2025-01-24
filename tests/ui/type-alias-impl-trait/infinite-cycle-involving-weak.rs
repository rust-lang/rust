//~ ERROR overflow normalizing the opaque type
#![feature(type_alias_impl_trait)]

type T = impl Copy;

static STATIC: T = None::<&'static T>;

fn main() {}
