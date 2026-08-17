//@ revisions: old next
//@[next] compile-flags: -Znext-solver

// Previously the fallback of param env normalization was elaborated param env
// when the normalization failed.
// The elaborated param env is entirely unnormalized which causes problems for
// places where we expect normalized param env, e.g. in lexical region solving.
//
// Now we map non-rigid aliases and unresolved infer vars to `Ty/Const/Region::Error`.

trait Trait1 {
    type Assoc1;
}

struct Indir;

trait Trait2 {
    type Assoc2;
}

impl<T> Trait2 for T
//~^ ERROR: the trait bound `Indir: Trait1` is not satisfied
where
    T: Trait1,
    T::Assoc1: 'static,
    T: Trait1<Assoc1 = <Indir as Trait1>::Assoc1>,
{
    type Assoc2 = ();
    //~^ ERROR: the trait bound `Indir: Trait1` is not satisfied
}

fn main() {}
