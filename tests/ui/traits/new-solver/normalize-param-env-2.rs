// check-pass
// compile-flags: -Znext-solver
// Issue 92505

trait A<T> {
    type I;

    fn f()
    where
        Self::I: A<T>,
    {
    }
}

impl<T> A<T> for () {
    type I = ();

    fn f()
    where
        Self::I: A<T>,
    {
        <() as A<T>>::f();
    }
}

fn main() {}
