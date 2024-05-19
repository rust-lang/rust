// Regression test for rust-lang/rust#63154
//
// Before, we would ICE after failing to normalize the destination type
// when checking call destinations and also when checking MIR
// assignment statements.

//@ check-pass

trait HasAssocType {
    type Inner;
}

impl HasAssocType for () {
    type Inner = ();
}

trait Tr<I, T>: Fn(I) -> Option<T> {}
impl<I, T, Q: Fn(I) -> Option<T>> Tr<I, T> for Q {}

fn f<T: HasAssocType>() -> impl Tr<T, T::Inner> {
    |_| None
}

fn g<T, Y>(f: impl Tr<T, Y>) -> impl Tr<T, Y> {
    f
}

fn h() {
    g(f())(());
}

fn main() {
    h();
}
