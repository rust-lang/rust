//@ compile-flags: -Znext-solver=coherence

#![recursion_limit = "10"]

trait Trait {}

struct W<T: ?Sized>(*const T);
trait TwoW {}
impl<T: ?Sized + TwoW> TwoW for W<W<T>> {}

impl<T: ?Sized + TwoW> Trait for W<T> {}
impl<T: ?Sized + TwoW> Trait for T {}
//~^ ERROR conflicting implementations of trait `Trait` for type `W

fn main() {}
