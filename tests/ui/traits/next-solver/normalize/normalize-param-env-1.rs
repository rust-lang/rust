//@ compile-flags: -Znext-solver

// Regression test for #108933.
// This was fixed by lazy norm of param env with the next solver.
// But it regressed again as we switched back to be consistent with
// the old solver. See #158643.

trait Add<Rhs> {
    type Sum;
}

impl Add<()> for () {
    type Sum = ();
}

type Unit = <() as Add<()>>::Sum;

trait Trait<C> {
    type Output;
}

fn f<T>()
where
    T: Trait<()>,
    <T as Trait<()>>::Output: Sized,
{
}

fn g<T>()
//~^ ERROR: the trait bound `T: Trait<()>` is not satisfied
where
    T: Trait<Unit>,
    <T as Trait<()>>::Output: Sized,
{
}

fn h<T>()
where
    T: Trait<()>,
    <T as Trait<Unit>>::Output: Sized,
{
}

fn main() {}
