// build-pass (FIXME(62277): could be check-pass?)
// pretty-expanded FIXME #23616

trait Trait<Input> {
    type Output;

    fn method() -> <Self as Trait<Input>>::Output;
}

impl<T> Trait<T> for () {
    type Output = ();

    fn method() {}
}

fn main() {}
