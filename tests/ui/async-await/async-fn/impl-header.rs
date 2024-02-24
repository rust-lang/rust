//@ edition:2018

struct F;

impl async Fn<()> for F {}
//~^ ERROR `async` trait implementations are unsupported
//~| ERROR the precise format of `Fn`-family traits' type parameters is subject to change
//~| ERROR manual implementations of `Fn` are experimental
//~| ERROR expected a `FnMut()` closure, found `F`
//~| ERROR not all trait items implemented, missing: `call`

fn main() {}
