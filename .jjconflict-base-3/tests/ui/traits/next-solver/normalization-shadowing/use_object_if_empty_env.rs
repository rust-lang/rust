//@ compile-flags: -Znext-solver
//@ check-pass

// Regression test for trait-system-refactor-initiative#244

trait Trait {
    type Assoc;
}

// We have param env candidate for the trait goal but not the projection.
// Under such circumstance, consider object candidate if the self_ty is trait object.
fn foo<T>(x: <dyn Trait<Assoc = T> as Trait>::Assoc) -> T
where
    dyn Trait<Assoc = T>: Trait,
{
    x
}

trait Id<'a> {
    type This: ?Sized;
}
impl<T: ?Sized> Id<'_> for T {
    type This = T;
}

// Ensure that we properly normalize alias self_ty before evaluating the goal.
fn alias_foo<T>(x: for<'a> fn(
    <<dyn Trait<Assoc = T> as Id<'a>>::This as Trait>::Assoc
)) -> fn(T)
where
    dyn Trait<Assoc = T>: Trait,
{
    x
}

fn main() {}
