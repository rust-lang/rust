//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally

trait AliasHaver {
    type Assoc;
}

trait Trait<'a> {}
impl<'a, T: 'a> Trait<'a> for T {}

// Constraints produced while normalizing implied bounds must reach lexical regionck.
trait Project {
    type Assoc;
}

impl<T: AliasHaver> Project for (T,)
where
    T::Assoc: for<'a> Trait<'a>,
{
    type Assoc = ();
}

struct Normalizes<T: Project>(T)
where
    T::Assoc: Clone;

trait TestTrait {}

impl<'a, T: AliasHaver> TestTrait for [Normalizes<(T,)>; 1]
//~^ ERROR: higher-ranked lifetime bound could not be satisfied
where
    T::Assoc: 'a,
{
}

fn main() {}
