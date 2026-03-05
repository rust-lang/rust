//@ compile-flags: -Znext-solver

// Regression test for <https://github.com/rust-lang/rust/issues/152716>.

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
