// build-pass (FIXME(62277): could be check-pass?)
// Check that associated types are `Sized`

// pretty-expanded FIXME #23616

trait Trait {
    type Output;

    fn is_sized(&self) -> Self::Output;
    fn wasnt_sized(&self) -> Self::Output { loop {} }
}

fn main() {}
