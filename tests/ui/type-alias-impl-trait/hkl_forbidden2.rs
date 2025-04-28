#![feature(type_alias_impl_trait)]

type Opaque<'a> = impl Sized + 'a;

trait Trait<'a> {
    type Assoc;
}

impl<'a> Trait<'a> for () {
    type Assoc = ();
}

#[define_opaque(Opaque)]
fn test() -> &'static dyn for<'a> Trait<'a, Assoc = Opaque<'a>> {
    &()
    //~^ ERROR: expected generic lifetime parameter, found `'a`
}

fn main() {}
