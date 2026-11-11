//@ compile-flags: -Znext-solver
//@ check-pass

// `(): Trait` is a global where-bound with a projection bound.
// This previously resulted in ambiguity as we considered both
// the impl and the where-bound while normalizing.

trait Trait {
    type Assoc;
}
impl Trait for () {
    type Assoc = &'static ();
}

fn foo<'a>(x: <() as Trait>::Assoc)
where
    (): Trait<Assoc = &'a ()>,
{
}

fn main() {}
