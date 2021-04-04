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
where
    Self::A: Baz,
    Self::B: Fiz,
{
    type A = ();
    //~^ ERROR overflow evaluating the requirement `<(T,) as Grault>::A == _`
    type B = bool;
    //~^ ERROR overflow evaluating the requirement `<(T,) as Grault>::A == _`
}
//~^^^^^^^^^^ ERROR overflow evaluating the requirement `<(T,) as Grault>::A == _`

fn main() {
    let x: <(_,) as Grault>::A = ();
}
