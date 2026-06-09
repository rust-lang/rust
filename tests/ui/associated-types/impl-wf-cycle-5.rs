//@ run-rustfix

#[allow(dead_code)]
trait Baz {}
impl Baz for () {}
impl<T> Baz for (T,) {}

#[allow(dead_code)]
trait Fiz {}
impl Fiz for bool {}

trait Grault {
    type A;
    type B;
}

impl Grault for () {
    type A = ();
    type B = bool;
}

impl<T> Grault for (T,)
//~^ ERROR overflow evaluating the requirement `<(T,) as Grault>::A == _`
where
    T: Grault,
    Self::A: Baz,
{
    type A = ();
    type B = bool;
}

fn main() {
    let _: <((),) as Grault>::A = ();
}
