//@ compile-flags: -Znext-solver

// Regression test for <https://github.com/rust-lang/rust/issues/152716>.
//
// This test checks that we hit the recursion limit for recursively defined projections.
// Normalization of `<T as Proj<'b>>::Assoc` could introduce the same projection again.
// Previously, we get into an infinite recursion.

trait Trait<T> {}
trait Proj<'a> {
    type Assoc;
}
fn foo<T>()
where
    T: for<'a> Proj<'a, Assoc = for<'b> fn(<T as Proj<'b>>::Assoc)>,
    (): Trait<<T as Proj<'static>>::Assoc>
    //~^ ERROR overflow evaluating the requirement `(): Trait<<T as Proj<'static>>::Assoc>` [E0275]
{
}

fn main() {}
