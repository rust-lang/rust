//@ compile-flags: -Znext-solver=globally -Zassumptions-on-binders

trait Trait<T> {}

trait Proj<'a> {
    type Assoc;
}

fn foo<'a, T>()
where
    T: Proj<'a, Assoc = fn(<T as Proj>::Assoc)>,
    (): Trait<<T as Proj<'a>>::Assoc>,
    //~^ ERROR the trait bound `(): Trait<fn(for<'a> fn(<T as Proj<'a>>::Assoc))>` is not satisfied
{
}

fn main() {}
