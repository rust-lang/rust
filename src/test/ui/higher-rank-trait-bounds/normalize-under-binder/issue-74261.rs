// check-pass

use std::marker::PhantomData;

trait A<'a> {
    type B;
    fn b(self) -> Self::B;
}

struct T;
struct S<'a>(PhantomData<&'a ()>);

impl<'a> A<'a> for T {
    type B = S<'a>;
    fn b(self) -> Self::B {
        S(PhantomData)
    }
}

fn s<TT, F>(t: TT, f: F)
where
    TT: for<'a> A<'a>,
    F: for<'a> FnOnce(<TT as A<'a>>::B)
{
    f(t.b());
}

fn main() {
    s(T, |_| {});
}
