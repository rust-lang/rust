pub struct B0;
pub struct B1;
use std::ops::Shl;
use std::ops::Sub;
pub type Shleft<A, B> = <A as Shl<B>>::Output;
pub type Sub1<A> = <A as Sub<B1>>::Output;
pub struct UInt<U, B> {
    pub(crate) msb: U,
    pub(crate) lsb: B,
}
impl<U, B, Ur, Br> Shl<UInt<Ur, Br>> for UInt<U, B>
where
    UInt<Ur, Br>: Sub<B1>,
    UInt<UInt<U, B>, B0>: Shl<Sub1<UInt<Ur, Br>>>,
{
    type Output = Shleft<UInt<UInt<U, B>, B0>, Sub1<UInt<Ur, Br>>>;
    fn shl(self, rhs: UInt<Ur, Br>) -> Self::Output {
        unimplemented!()
    }
}
