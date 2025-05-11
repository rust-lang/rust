#![feature(type_alias_impl_trait)]

pub trait Trait<'a> {
    type Assoc;
}

trait Test<'a> {}

pub type Foo = impl for<'a> Trait<'a, Assoc = impl Test<'a>>;
//~^ ERROR `impl Trait` cannot capture higher-ranked lifetime from outer `impl Trait`
//~| ERROR `impl Trait` cannot capture higher-ranked lifetime from outer `impl Trait`

impl Trait<'_> for () {
    type Assoc = ();
}

impl Test<'_> for () {}

#[define_opaque(Foo)]
fn constrain() -> Foo {
    ()
}

fn main() {}
