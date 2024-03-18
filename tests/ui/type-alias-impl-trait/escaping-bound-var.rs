#![feature(type_alias_impl_trait)]

pub trait Trait<'a> {
    type Assoc;
}

trait Test<'a> {}

pub type Foo = impl for<'a> Trait<'a, Assoc = impl Test<'a>>;
//~^ ERROR `impl Trait` cannot capture higher-ranked lifetime from outer `impl Trait`

impl Trait<'_> for () {
    type Assoc = ();
}

impl Test<'_> for () {}

fn constrain() -> Foo {
    ()
    //~^ ERROR expected generic lifetime parameter, found `'static`
    // FIXME(aliemjay): Undesirable error message appears because error regions
    // are converterted internally into `'?0` which corresponds to `'static`
    // This should be fixed in a later commit.
}

fn main() {}
