// Regression test for #52873. We used to ICE due to unexpected
// overflows when checking for "blanket impl inclusion".

use std::marker::PhantomData;
use std::cmp::Ordering;
use std::ops::{Add, Mul};

pub type True = B1;
pub type False = B0;
pub type U0 = UTerm;
pub type U1 = UInt<UTerm, B1>;

pub trait NonZero {}

pub trait Bit {
}

pub trait Unsigned {
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Hash, Debug, Default)]
pub struct B0;

impl B0 {
    #[inline]
    pub fn new() -> B0 {
        B0
    }
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Hash, Debug, Default)]
pub struct B1;

impl B1 {
    #[inline]
    pub fn new() -> B1 {
        B1
    }
}

impl Bit for B0 {
}

impl Bit for B1 {
}

impl NonZero for B1 {}

pub trait PrivatePow<Y, N> {
    type Output;
}
pub type PrivatePowOut<A, Y, N> = <A as PrivatePow<Y, N>>::Output;

pub type Add1<A> = <A as Add<::B1>>::Output;
pub type Prod<A, B> = <A as Mul<B>>::Output;
pub type Square<A> = <A as Mul>::Output;
pub type Sum<A, B> = <A as Add<B>>::Output;

#[derive(Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Hash, Debug, Default)]
pub struct UTerm;

impl UTerm {
    #[inline]
    pub fn new() -> UTerm {
        UTerm
    }
}

impl Unsigned for UTerm {
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Hash, Debug, Default)]
pub struct UInt<U, B> {
    _marker: PhantomData<(U, B)>,
}

impl<U: Unsigned, B: Bit> UInt<U, B> {
    #[inline]
    pub fn new() -> UInt<U, B> {
        UInt {
            _marker: PhantomData,
        }
    }
}

impl<U: Unsigned, B: Bit> Unsigned for UInt<U, B> {
}

impl<U: Unsigned, B: Bit> NonZero for UInt<U, B> {}

impl Add<B0> for UTerm {
    type Output = UTerm;
    fn add(self, _: B0) -> Self::Output {
        UTerm
    }
}

impl<U: Unsigned, B: Bit> Add<B0> for UInt<U, B> {
    type Output = UInt<U, B>;
    fn add(self, _: B0) -> Self::Output {
        UInt::new()
    }
}

impl<U: Unsigned> Add<U> for UTerm {
    type Output = U;
    fn add(self, _: U) -> Self::Output {
        #[allow(deprecated)]
        unsafe { ::std::mem::uninitialized() }
    }
}

impl<U: Unsigned, B: Bit> Mul<B0> for UInt<U, B> {
    type Output = UTerm;
    fn mul(self, _: B0) -> Self::Output {
        UTerm
    }
}

impl<U: Unsigned, B: Bit> Mul<B1> for UInt<U, B> {
    type Output = UInt<U, B>;
    fn mul(self, _: B1) -> Self::Output {
        UInt::new()
    }
}

impl<U: Unsigned> Mul<U> for UTerm {
    type Output = UTerm;
    fn mul(self, _: U) -> Self::Output {
        UTerm
    }
}

impl<Ul: Unsigned, B: Bit, Ur: Unsigned> Mul<UInt<Ur, B>> for UInt<Ul, B0>
where
    Ul: Mul<UInt<Ur, B>>,
{
    type Output = UInt<Prod<Ul, UInt<Ur, B>>, B0>;
    fn mul(self, _: UInt<Ur, B>) -> Self::Output {
        unsafe { ::std::mem::uninitialized() }
    }
}

pub trait Pow<Exp> {
    type Output;
}

impl<X: Unsigned, N: Unsigned> Pow<N> for X
where
    X: PrivatePow<U1, N>,
{
    type Output = PrivatePowOut<X, U1, N>;
}

impl<Y: Unsigned, X: Unsigned> PrivatePow<Y, U0> for X {
    type Output = Y;
}

impl<Y: Unsigned, X: Unsigned> PrivatePow<Y, U1> for X
where
    X: Mul<Y>,
{
    type Output = Prod<X, Y>;
}

impl<Y: Unsigned, U: Unsigned, B: Bit, X: Unsigned> PrivatePow<Y, UInt<UInt<U, B>, B0>> for X
where
    X: Mul,
    Square<X>: PrivatePow<Y, UInt<U, B>>,
{
    type Output = PrivatePowOut<Square<X>, Y, UInt<U, B>>;
}
