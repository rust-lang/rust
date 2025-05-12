// Regression test for #79714

trait Grault {
    type A;
}

impl<T: Grault> Grault for (T,)
//~^ ERROR overflow evaluating the requirement `<(T,) as Grault>::A == _`
where
    Self::A: Copy,
{
    type A = ();
}

fn main() {}
