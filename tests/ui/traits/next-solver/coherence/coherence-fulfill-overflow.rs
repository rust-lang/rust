//@ compile-flags: -Znext-solver=coherence

#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]
#![recursion_limit = "10"]

trait Trait {}

struct W<T>(*const T);
trait TwoW {}
impl<T: TwoW> TwoW for W<W<T>> {}

impl<T: TwoW> Trait for W<T> {}
impl<T: TwoW> Trait for T {}
//~^ ERROR conflicting implementations of trait `Trait` for type `W

fn main() {}
