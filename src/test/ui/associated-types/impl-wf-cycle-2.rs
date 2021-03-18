// Regression test for #79714

trait Grault {
    type A;
}

impl<T: Grault> Grault for (T,)
where
    Self::A: Copy,
{
    type A = ();
    //~^ ERROR overflow evaluating the requirement `<(T,) as Grault>::A == _`
}
//~^^^^^^^ ERROR overflow evaluating the requirement `<(T,) as Grault>::A == _`

fn main() {}
