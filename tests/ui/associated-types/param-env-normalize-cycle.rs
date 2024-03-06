// Minimized case from typenum that didn't compile because:
// - We tried to normalize the ParamEnv of the second impl
// - This requires trying to normalize `GrEq<Self, Square<Square<U>>>`
// - This requires proving `Square<Square<U>>: Sized` so that the first impl
//   applies
// - This requires Providing `Square<Square<U>>` is well-formed, so that we
//   can use the `Sized` bound on `Mul::Output`
// - This requires proving `Square<U>: Mul`
// - But first we tried normalizing the whole obligation, including the
//   ParamEnv, which leads to a cycle error.

//@ check-pass

trait PrivateSquareRoot {}

pub trait Mul<Rhs = Self> {
    type Output;
}

pub trait IsGreaterOrEqual<Rhs> {
    type Output;
}

pub type Square<A> = <A as Mul>::Output;
pub type GrEq<A, B> = <A as IsGreaterOrEqual<B>>::Output;

impl<A, B> IsGreaterOrEqual<B> for A {
    type Output = ();
}

impl<U> PrivateSquareRoot for U
where
    U: Mul,
    Square<U>: Mul,
    GrEq<Self, Square<Square<U>>>: Sized,
{
}

fn main() {}
