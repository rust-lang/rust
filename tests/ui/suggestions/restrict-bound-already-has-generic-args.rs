// Regression test for https://github.com/rust-lang/rust/issues/142803.

trait Pair {
    type Left;
    type Right;

    fn split(self) -> (Self::Left, Self::Right);
}

impl<A, B> Pair for (A, B) {
    type Left = A;
    type Right = B;

    fn split(self) -> (Self::Left, Self::Right) {
        self
    }
}

fn frob<A, B>(pair: impl Pair<Left = A>) -> impl Pair<Left = A, Right = B> {
    //~^ ERROR type mismatch
    //~| HELP consider further restricting this bound
    //~| SUGGESTION , Right = B
    let (left, right) = pair.split();
    (left, right)
}

fn main() {}
