// Regression test for #79714

trait Baz {}
impl Baz for () {}
impl<T> Baz for (T,) {}

trait Fiz {}
impl Fiz for bool {}

trait Grault {
    type A;
    type B;
}

impl<T: Grault> Grault for (T,)
//~^ ERROR overflow evaluating the requirement `<(T,) as Grault>::A == _`
where
    Self::A: Baz,
    Self::B: Fiz,
{
    type A = ();
    type B = bool;
}

fn main() {
    let x: <(_,) as Grault>::A = ();
}
