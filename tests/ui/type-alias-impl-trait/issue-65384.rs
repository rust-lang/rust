#![feature(type_alias_impl_trait)]

trait MyTrait {}

impl MyTrait for () {}

type Bar = impl MyTrait;

impl MyTrait for Bar {}
//~^ ERROR: conflicting implementations of trait `MyTrait` for type `()`

#[define_opaque(Bar)]
fn bazr() -> Bar {}

fn main() {}
