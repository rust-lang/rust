// check-pass
// compile-flags: -Ztrait-solver=next
// Issue 108933

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
