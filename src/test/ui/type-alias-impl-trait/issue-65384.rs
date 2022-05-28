#![feature(type_alias_impl_trait)]
#![allow(incomplete_features)]

trait MyTrait {}

impl MyTrait for () {}

type Bar = impl MyTrait;

impl MyTrait for Bar {}
//~^ ERROR: cannot implement trait on type alias impl trait

fn bazr() -> Bar { }

fn main() {}
