//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait With {
    type Assoc;
}
impl<T> With for T {
    type Assoc = T;
}

trait Trait {}
fn foo<T>()
where
    T: With<Assoc = u32>,
    // This where-bound only global after normalization. We still
    // check whether it is a trivial bound.
    T::Assoc: Trait,
    //~^ ERROR the trait bound `u32: Trait` is not satisfied
{
}

fn main() {}
