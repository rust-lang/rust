//@ compile-flags: -Znext-solver=globally -Zassumptions-on-binders

trait Trait<T> {}

trait Proj<'a> {
    type Assoc;
}

fn foo<'a, T>()
where
    T: Proj<'a, Assoc = fn(<T as Proj>::Assoc)>,
    (): Trait<<T as Proj<'a>>::Assoc>,
    //~^ ERROR overflow evaluating the requirement `(): Trait<<T as Proj<'a>>::Assoc>`
{
}

fn main() {}
