trait T0<'a, A> {
    type O;
}

struct L<T> {
    f: T,
}

// explicitly named variants of what one would normally denote by the
// unit type `()`. Why do this? So that we can differentiate them in
// the diagnostic output.
struct Unit1;
struct Unit2;
struct Unit3;
struct Unit4;

impl<'a, A, T> T0<'a, A> for L<T>
where
    T: FnMut(A) -> Unit3,
{
    type O = T::Output;
}

trait T1: for<'r> Ty<'r> {
    fn m<'a, B: Ty<'a>, F>(&self, f: F) -> Unit1
    where
        F: for<'r> T0<'r, (<Self as Ty<'r>>::V,), O = <B as Ty<'r>>::V>,
    {
        unimplemented!();
    }
}

trait Ty<'a> {
    type V;
}

fn main() {
    let v = Unit2.m(
        L {
            //~^ ERROR to be a closure that returns `Unit3`, but it returns `Unit4`
            //~| ERROR type mismatch
            f: |x| {
                drop(x);
                Unit4
            },
        },
    );
}

impl<'a> Ty<'a> for Unit2 {
    type V = &'a u8;
}

impl T1 for Unit2 {}
