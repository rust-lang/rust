#![feature(type_alias_impl_trait)]

pub trait Trait<'a> {
    type Assoc;
}

trait Test<'a> {}

pub type Foo = impl for<'a> Trait<'a, Assoc = impl Test<'a>>;
//~^ ERROR cannot capture late-bound lifetime in type alias impl trait

impl Trait<'_> for () {
    type Assoc = ();
}

impl Test<'_> for () {}

fn constrain() -> Foo {
    ()
}

fn main() {}
