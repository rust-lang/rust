#![feature(rustc_attrs)]

#[rustc_strict_coherence]
trait Foo {}
//~^ ERROR to use `strict_coherence` on this trait, the `with_negative_coherence` feature must be enabled

fn main() {}
