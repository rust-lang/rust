//@ check-pass
// Check that associated types are `Sized`


trait Trait {
    type Output;

    fn is_sized(&self) -> Self::Output;
    fn wasnt_sized(&self) -> Self::Output { loop {} }
}

fn main() {}
