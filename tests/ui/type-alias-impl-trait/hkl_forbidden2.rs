#![feature(type_alias_impl_trait)]
//~^ ERROR: expected generic lifetime parameter, found `'a`

type Opaque<'a> = impl Sized + 'a;

trait Trait<'a> {
    type Assoc;
}

impl<'a> Trait<'a> for () {
    type Assoc = ();
}

fn test() -> &'static dyn for<'a> Trait<'a, Assoc = Opaque<'a>> {
    &()
}

fn main() {}
