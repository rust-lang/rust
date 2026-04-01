//@ compile-flags: -Znext-solver
//@ check-pass

// Regression test for trait-system-refactor-initiative#105. We previously encountered
// an ICE in typenum as `forced_ambiguity` failed. While this test no longer causes
// `forced_ambiguity` to error, we still want to use it as a regression test.

pub struct UInt<U, B> {
    _msb: U,
    _lsb: B,
}
pub struct B1;
pub trait Sub<Rhs> {
    type Output;
}
impl<U, B> Sub<B1> for UInt<UInt<U, B>, B1> {
    type Output = ();
}
impl<U> Sub<B1> for UInt<U, ()>
where
    U: Sub<B1>,
    U::Output: Send,
{
    type Output = ();
}

pub trait Op<N, R, I> {
    fn op(&self) {
        unimplemented!()
    }
}
trait OpIf<N, R, I> {}

impl<N, Ur, Br, I> Op<N, UInt<Ur, Br>, I> for ()
where
    N: Sub<I>,
    (): OpIf<N, UInt<UInt<Ur, Br>, N::Output>, I>,
{
}
impl<N, R, Ui, Bi> OpIf<N, R, UInt<Ui, Bi>> for ()
where
    UInt<Ui, Bi>: Sub<B1>,
    (): Op<N, R, <UInt<Ui, Bi> as Sub<B1>>::Output>,
{
}
impl<N, R> OpIf<N, R, ()> for () where R: Sub<N> {}

pub trait Compute {
    type Output;
}

pub fn repro<Ul, Bl>()
where
    UInt<Ul, Bl>: Compute,
    <UInt<Ul, Bl> as Compute>::Output: Sub<B1>,
    (): Op<UInt<(), Bl>, (), ()>,
{
    ().op();
}
fn main() {}
