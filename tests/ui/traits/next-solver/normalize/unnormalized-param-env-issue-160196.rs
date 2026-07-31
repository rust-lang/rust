//@ compile-flags: -Znext-solver
//@ check-fail

trait Trait1 {
    type Assoc1;
}

struct Indir;

trait Trait2 {
    type Assoc2;
}

impl<T> Trait2 for T
//~^ ERROR the trait bound `Indir: Trait1` is not satisfied
//~| ERROR the trait bound `T: Trait2` is not satisfied
where
    T: Trait1,
    T::Assoc1: 'static,
    T: Trait1<Assoc1 = <Indir as Trait1>::Assoc1>,
    //~^ ERROR the trait bound `Indir: Trait1` is not satisfied
{
    type Assoc2 = ();
    //~^ ERROR the trait bound `Indir: Trait1` is not satisfied
    //~| ERROR the trait bound `T: Trait2` is not satisfied
}

fn main() {}
