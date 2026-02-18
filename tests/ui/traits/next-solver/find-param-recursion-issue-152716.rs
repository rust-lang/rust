//@ compile-flags: -Znext-solver
//~^ ERROR overflow normalizing the associated type `<T as Proj<'b>>::Assoc` [E0275]

// Regression test for <https://github.com/rust-lang/rust/issues/152716>.

trait Trait<T> {}
trait Proj<'a> {
    type Assoc;
}
fn foo<T>()
where
    T: for<'a> Proj<'a, Assoc = for<'b> fn(<T as Proj<'b>>::Assoc)>,
    (): Trait<<T as Proj<'static>>::Assoc>
{
}

fn main() {}
