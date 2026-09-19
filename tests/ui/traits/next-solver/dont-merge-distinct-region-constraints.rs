//@ compile-flags: -Znext-solver

trait Trait<'a> {}

fn needs_trait<'a, T: Trait<'a>>() {}

fn test<'a, 'b, T>()
where
    T: Trait<'a>,
    T: Trait<'b>,
{
    // Both candidates constrain the inferred lifetime, but neither is
    // unconstrained. Merging them would require representing the disjunction
    // that the inferred lifetime is either `'a` or `'b`.
    needs_trait::<T>();
    //~^ ERROR type annotations needed: cannot satisfy `T: Trait<'_>`
}

fn main() {}
