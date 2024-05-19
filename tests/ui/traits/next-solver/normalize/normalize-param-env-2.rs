//@ compile-flags: -Znext-solver
//@ known-bug: #92505

// When checking that the impl method where-bounds are implied by the trait,
// we prove  `<() as A<T>>::Assoc: A<T>` in the environment `<() as A<T>>::Assoc: A<T>`.
//
// Normalizing `<() as A<T>>::Assoc` is ambiguous in that environment. The
// where-bound `<() as A<T>>::Assoc: A<T>` may apply, resulting in overflow.
trait A<T> {
    type Assoc;

    fn f()
    where
        Self::Assoc: A<T>,
    {
    }
}

impl<T> A<T> for () {
    type Assoc = ();

    fn f()
    where
        Self::Assoc: A<T>,
    {

        <() as A<T>>::f();
    }
}

fn main() {}
