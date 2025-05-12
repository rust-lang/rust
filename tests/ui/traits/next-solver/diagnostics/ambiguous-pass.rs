//@ compile-flags: -Znext-solver

trait Trait {
    type Assoc;
}

struct W<T>(*mut T);
impl<T> Trait for W<W<T>>
where
    W<T>: Trait,
{
    type Assoc = ();
}

trait NoOverlap {}
impl<T: Trait> NoOverlap for T {}

impl<T: Trait<Assoc = ()>> NoOverlap for W<T> {}
//~^ ERROR conflicting implementations of trait `NoOverlap` for type `W<_>`

fn main() {}
