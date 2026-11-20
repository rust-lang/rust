//@ check-pass
// https://github.com/rust-lang/rust/issues/119792

struct Wrapper<T>(T);

trait Div<Rhs> {}
trait Mul<Rhs> {
    type Output;
}

impl<T> Mul<T> for Wrapper<T> {
    type Output = ();
}

impl<T> Div<Self> for Wrapper<T> {}

pub trait NumOps<Rhs> {}

impl<T, Rhs> NumOps<Rhs> for T where T: Mul<Rhs, Output = ()> + Div<Rhs> {}
