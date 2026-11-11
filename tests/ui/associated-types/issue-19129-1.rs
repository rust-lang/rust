//@ check-pass

trait Trait<Input> {
    type Output;

    fn method() -> <Self as Trait<Input>>::Output;
}

impl<T> Trait<T> for () {
    type Output = ();

    fn method() {}
}

fn main() {}
