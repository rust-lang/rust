//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally

trait Trait<T> {}

trait Proj<'a> {
    type Assoc;
}

fn foo<'a, T>()
//~^ ERROR overflow evaluating the requirement `<T as Proj<'a>>::Assoc == _`
where
    T: Proj<'a, Assoc = fn(<T as Proj>::Assoc)>,
    (): Trait<<T as Proj<'a>>::Assoc>,
{
}

fn main() {}
