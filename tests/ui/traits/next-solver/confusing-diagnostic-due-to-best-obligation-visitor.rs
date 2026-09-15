//@compile-flags: -Znext-solver=globally
//@check-fail

// See issue https://github.com/rust-lang/rust/issues/161882, test makes sure we don't report
// irrelevant and not implemented bounds when we can't proof an obligation.

struct MyError;

trait MaybeFallible {
    type Error: MaybeError; //~ ERROR: the trait bound `MyError: From<<Self as MaybeFallible>::Error>` is not satisfied
}

trait MaybeError: Sized
where
    MyError: From<Self>,
{
    type ComposeWithOther: MaybeError; //~ ERROR: the trait bound `MyError: From<<Self as MaybeError>::ComposeWithOther>` is not satisfied
}

fn compose<A: MaybeFallible, B>() -> Result<(), ()>
where
    <A::Error as MaybeError>::ComposeWithOther: From<B>, //~ ERROR: the trait bound `MyError: From<<A as MaybeFallible>::Error>` is not satisfied
{
    todo!()
}

fn main() {}
