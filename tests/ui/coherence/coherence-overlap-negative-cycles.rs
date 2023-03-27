#![feature(trivial_bounds)]
#![feature(negative_impls)]
#![feature(rustc_attrs)]
#![feature(with_negative_coherence)]
#![allow(trivial_bounds)]

#[rustc_strict_coherence]
trait MyTrait {}

struct Foo {}

impl !MyTrait for Foo {}

impl MyTrait for Foo where Foo: MyTrait {}
//~^ ERROR: conflicting implementations of trait `MyTrait`

fn main() {}
