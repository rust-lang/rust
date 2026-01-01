//@ compile-flags: -Znext-solver
//@ check-pass

// A regression test for https://github.com/rust-lang/trait-system-refactor-initiative/issues/252.
// `fn fudge_inference_if_ok` might lose relationships between ty vars so we need to normalize
// them inside the fudge scope.

pub struct Error;

trait Throw {
    type Error;
    fn from_error(error: Self::Error) -> Self;
}
impl<T, E> Throw for Result<T, E> {
    type Error = E;
    fn from_error(_: Self::Error) -> Self {
        unimplemented!()
    }
}

fn op<F, T>(_: F) -> Result<T, Error>
where
    F: FnOnce() -> Result<T, Error>,
{
    unimplemented!()
}
pub fn repro() -> Result<(), Error> {
    op(|| Throw::from_error(Error))?
}

fn main() {}
