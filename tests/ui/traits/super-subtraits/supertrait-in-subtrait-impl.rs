#![feature(rustc_attrs)]

trait Sup {
    type A;
}

#[rustc_supertrait_in_subtrait_impl]
trait Sub: Sup {}

struct S;
impl Sub for S {
    //~^ ERROR: the trait bound `S: Sup` is not satisfied
    type A = ();
    //~^ ERROR: the trait bound `S: Sup` is not satisfied
}

fn main() {}
