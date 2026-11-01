//@ compile-flags: -Znext-solver
//@ check-pass

// Regression test for trait-system-refactor-initiative#246
// There're some smaller intermediate minimizations in the issue comments
// but they may not catch the same problem as in the full version.
//
// Fixed by eager norm and marking param env as rigid. See #158643.

use std::ops::{BitAnd, BitOr, BitXor, Neg, Not, Shl, Shr};
pub struct B0;

impl Clone for B0 {
    fn clone(&self) -> B0 {
        loop {}
    }
}

impl Copy for B0 {}

impl Default for B0 {
    fn default() -> B0 {
        loop {}
    }
}

pub struct B1;

impl Clone for B1 {
    fn clone(&self) -> B1 {
        loop {}
    }
}

impl Copy for B1 {}

impl Default for B1 {
    fn default() -> B1 {
        loop {}
    }
}

impl Bit for B0 {
    const U8: u8 = 0;
    const BOOL: bool = false;
}
impl Bit for B1 {
    const U8: u8 = 1;
    const BOOL: bool = true;
}

pub type U0 = UTerm;
pub type U1 = UInt<UTerm, B1>;

pub type U2 = UInt<UInt<UTerm, B1>, B0>;

pub trait NonZero {}

pub trait Ord {}
pub trait Bit: Copy + Default + 'static {
    const U8: u8;
    const BOOL: bool;
}
pub trait Unsigned: Copy + Default + 'static {
}

pub type Shleft<A, B> = <A as Shl<B>>::Output;

pub type Sum<A, B> = <A as Add<B>>::Output;
pub type Diff<A, B> = <A as Sub<B>>::Output;
pub type Prod<A, B> = <A as Mul<B>>::Output;
pub type Quot<A, B> = <A as Div<B>>::Output;

pub type Gcf<A, B> = <A as Gcd<B>>::Output;
pub type Add1<A> = <A as Add<B1>>::Output;
pub type Sub1<A> = <A as Sub<B1>>::Output;

pub type Compare<A, B> = <A as Cmp<B>>::Output;
pub type Length<T> = <T as Len>::Output;

pub type Minimum<A, B> = <A as Min<B>>::Output;
pub type Maximum<A, B> = <A as Max<B>>::Output;

pub trait Trim {
    type Output;
}
pub type TrimOut<A> = <A as Trim>::Output;
pub trait TrimTrailingZeros {
    type Output;
}

pub trait Invert {
    type Output;
}

pub trait PrivateInvert<Rhs> {
    type Output;
}
pub type PrivateInvertOut<A, Rhs> = <A as PrivateInvert<Rhs>>::Output;
pub struct InvertedUTerm;
pub struct InvertedUInt<IU, B: Bit> {
    msb: IU,
    lsb: B,
}

pub trait PrivateSub<Rhs = Self> {
    type Output;
}
pub type PrivateSubOut<A, Rhs> = <A as PrivateSub<Rhs>>::Output;

pub trait InvertedUnsigned {}
impl InvertedUnsigned for InvertedUTerm {}
impl<IU, B: Bit> InvertedUnsigned for InvertedUInt<IU, B> {}

impl<U: Unsigned, B: Bit> Invert for UInt<U, B>
where
    U: PrivateInvert<InvertedUInt<InvertedUTerm, B>>,
{
    type Output = PrivateInvertOut<U, InvertedUInt<InvertedUTerm, B>>;
}
impl<IU> PrivateInvert<IU> for UTerm {
    type Output = IU;
}
impl<IU: InvertedUnsigned, U: Unsigned, B: Bit> PrivateInvert<IU> for UInt<U, B>
where
    U: PrivateInvert<InvertedUInt<IU, B>>,
{
    type Output = PrivateInvertOut<U, InvertedUInt<IU, B>>;
}
impl Invert for InvertedUTerm {
    type Output = UTerm;
}
impl<IU: InvertedUnsigned, B: Bit> Invert for InvertedUInt<IU, B>
where
    IU: PrivateInvert<UInt<UTerm, B>>,
{
    type Output = <IU as PrivateInvert<UInt<UTerm, B>>>::Output;
}
impl<U: Unsigned> PrivateInvert<U> for InvertedUTerm {
    type Output = U;
}
impl<U: Unsigned, IU, B: Bit> PrivateInvert<U> for InvertedUInt<IU, B>
where
    IU: PrivateInvert<UInt<U, B>>,
{
    type Output = <IU as PrivateInvert<UInt<U, B>>>::Output;
}
impl TrimTrailingZeros for InvertedUTerm {
    type Output = InvertedUTerm;
}
impl<IU> TrimTrailingZeros for InvertedUInt<IU, B1> {
    type Output = Self;
}
impl<IU: InvertedUnsigned> TrimTrailingZeros for InvertedUInt<IU, B0>
where
    IU: TrimTrailingZeros,
{
    type Output = <IU as TrimTrailingZeros>::Output;
}
impl<U: Unsigned> Trim for U
where
    U: Invert,
    <U as Invert>::Output: TrimTrailingZeros,
    <<U as Invert>::Output as TrimTrailingZeros>::Output: Invert,
{
    type Output = <<<U as Invert>::Output as TrimTrailingZeros>::Output as Invert>::Output;
}
pub trait PrivateCmp<Rhs, SoFar> {
    type Output;
}
pub type PrivateCmpOut<A, Rhs, SoFar> = <A as PrivateCmp<Rhs, SoFar>>::Output;
pub trait PrivateSetBit<I, B> {
    type Output;
}
pub type PrivateSetBitOut<N, I, B> = <N as PrivateSetBit<I, B>>::Output;
pub trait PrivateDiv<N, D, Q, R, I> {
    type Quotient;
    type Remainder;
}
pub type PrivateDivQuot<N, D, Q, R, I> = <() as PrivateDiv<N, D, Q, R, I>>::Quotient;
pub type PrivateDivRem<N, D, Q, R, I> = <() as PrivateDiv<N, D, Q, R, I>>::Remainder;
pub trait PrivateDivIf<N, D, Q, R, I, RcmpD> {
    type Quotient;
    type Remainder;
}
pub type PrivateDivIfQuot<N, D, Q, R, I, RcmpD> =
    <() as PrivateDivIf<N, D, Q, R, I, RcmpD>>::Quotient;
pub type PrivateDivIfRem<N, D, Q, R, I, RcmpD> =
    <() as PrivateDivIf<N, D, Q, R, I, RcmpD>>::Remainder;

pub trait PrivateMin<Rhs, CmpResult> {
    type Output;
}
pub type PrivateMinOut<A, B, CmpResult> = <A as PrivateMin<B, CmpResult>>::Output;
pub trait PrivateMax<Rhs, CmpResult> {
    type Output;
}
pub type PrivateMaxOut<A, B, CmpResult> = <A as PrivateMax<B, CmpResult>>::Output;

pub trait Cmp<Rhs = Self> {
    type Output;
}

pub trait Len {
    type Output;
}

pub trait Min<Rhs = Self> {
    type Output;
}
pub trait Max<Rhs = Self> {
    type Output;
}

pub trait Gcd<Rhs> {
    type Output;
}

pub struct UTerm;

impl Clone for UTerm {
    fn clone(&self) -> UTerm {
        loop {}
    }
}

impl Copy for UTerm {}

impl Default for UTerm {
    fn default() -> UTerm {
        loop {}
    }
}

impl Unsigned for UTerm {
}
pub struct UInt<U, B> {
    pub(crate) msb: U,
    pub(crate) lsb: B,
}

impl<U, B> Clone for UInt<U, B> {
    fn clone(&self) -> UInt<U, B> {
        loop {}
    }
}

impl<U: Copy, B: Copy> Copy for UInt<U, B> {}

impl<U, B> Default for UInt<U, B> {
    fn default() -> UInt<U, B> {
        loop {}
    }
}

impl<U: Unsigned, B: Bit> Unsigned for UInt<U, B> {
}
impl<U: Unsigned, B: Bit> NonZero for UInt<U, B> {}

impl Len for UTerm {
    type Output = U0;
}
impl<U: Unsigned, B: Bit> Len for UInt<U, B>
where
    U: Len,
    Length<U>: Add<B1>,
    Add1<Length<U>>: Unsigned,
{
    type Output = Add1<Length<U>>;
}

impl Add<B1> for UTerm {
    type Output = UInt<UTerm, B1>;
    fn add(self, _: B1) -> Self::Output {
        loop {}
    }
}

impl<U: Unsigned> Add<B1> for UInt<U, B1>
where
    U: Add<B1>,
    Add1<U>: Unsigned,
{
    type Output = UInt<Add1<U>, B0>;
    fn add(self, _: B1) -> Self::Output {
        loop {}
    }
}
impl<U: Unsigned> Add<U> for UTerm {
    type Output = U;
    fn add(self, rhs: U) -> Self::Output {
        loop {}
    }
}

impl<Ul: Unsigned, Ur: Unsigned> Add<UInt<Ur, B1>> for UInt<Ul, B0>
where
    Ul: Add<Ur>,
{
    type Output = UInt<Sum<Ul, Ur>, B1>;
    fn add(self, rhs: UInt<Ur, B1>) -> Self::Output {
        loop {}
    }
}

impl Sub<B1> for UInt<UTerm, B1> {
    type Output = UTerm;
    fn sub(self, _: B1) -> Self::Output {
        loop {}
    }
}
impl<U: Unsigned> Sub<B1> for UInt<U, B0>
where
    U: Sub<B1>,
    Sub1<U>: Unsigned,
{
    type Output = UInt<Sub1<U>, B1>;
    fn sub(self, _: B1) -> Self::Output {
        loop {}
    }
}

impl<Ul: Unsigned, Bl: Bit, Ur: Unsigned> Sub<Ur> for UInt<Ul, Bl>
where
    UInt<Ul, Bl>: PrivateSub<Ur>,
    PrivateSubOut<UInt<Ul, Bl>, Ur>: Trim,
{
    type Output = TrimOut<PrivateSubOut<UInt<Ul, Bl>, Ur>>;
    fn sub(self, rhs: Ur) -> Self::Output {
        loop {}
    }
}
impl<U: Unsigned> PrivateSub<UTerm> for U {
    type Output = U;
}

impl<Ul: Unsigned, Ur: Unsigned> PrivateSub<UInt<Ur, B1>> for UInt<Ul, B1>
where
    Ul: PrivateSub<Ur>,
{
    type Output = UInt<PrivateSubOut<Ul, Ur>, B0>;
}

impl<U: Unsigned, B: Bit> Shl<UTerm> for UInt<U, B> {
    type Output = UInt<U, B>;
    fn shl(self, _: UTerm) -> Self::Output {
        loop {}
    }
}

impl<U: Unsigned, B: Bit, Ur: Unsigned, Br: Bit> Shl<UInt<Ur, Br>> for UInt<U, B>
where
    UInt<Ur, Br>: Sub<B1>,
    UInt<UInt<U, B>, B0>: Shl<Sub1<UInt<Ur, Br>>>,
{
    type Output = Shleft<UInt<UInt<U, B>, B0>, Sub1<UInt<Ur, Br>>>;
    fn shl(self, rhs: UInt<Ur, Br>) -> Self::Output {
        loop {}
    }
}

impl<U: Unsigned> Mul<U> for UTerm {
    type Output = UTerm;
    fn mul(self, _: U) -> Self::Output {
        loop {}
    }
}
impl<Ul: Unsigned, B: Bit, Ur: Unsigned> Mul<UInt<Ur, B>> for UInt<Ul, B0>
where
    Ul: Mul<UInt<Ur, B>>,
{
    type Output = UInt<Prod<Ul, UInt<Ur, B>>, B0>;
    fn mul(self, rhs: UInt<Ur, B>) -> Self::Output {
        loop {}
    }
}
impl<Ul: Unsigned, B: Bit, Ur: Unsigned> Mul<UInt<Ur, B>> for UInt<Ul, B1>
where
    Ul: Mul<UInt<Ur, B>>,
    UInt<Prod<Ul, UInt<Ur, B>>, B0>: Add<UInt<Ur, B>>,
{
    type Output = Sum<UInt<Prod<Ul, UInt<Ur, B>>, B0>, UInt<Ur, B>>;
    fn mul(self, rhs: UInt<Ur, B>) -> Self::Output {
        loop {}
    }
}

impl<U: Unsigned, B: Bit> Cmp<UInt<U, B>> for UTerm {
    type Output = Less;
}
impl<Ul: Unsigned, Ur: Unsigned> Cmp<UInt<Ur, B0>> for UInt<Ul, B0>
where
    Ul: PrivateCmp<Ur, Equal>,
{
    type Output = PrivateCmpOut<Ul, Ur, Equal>;
}
impl<Ul: Unsigned, Ur: Unsigned> Cmp<UInt<Ur, B1>> for UInt<Ul, B1>
where
    Ul: PrivateCmp<Ur, Equal>,
{
    type Output = PrivateCmpOut<Ul, Ur, Equal>;
}

impl<Ul: Unsigned, Ur: Unsigned> Cmp<UInt<Ur, B0>> for UInt<Ul, B1>
where
    Ul: PrivateCmp<Ur, Greater>,
{
    type Output = PrivateCmpOut<Ul, Ur, Greater>;
}

impl<Ul, Ur, SoFar> PrivateCmp<UInt<Ur, B1>, SoFar> for UInt<Ul, B1>
where
    Ul: Unsigned,
    Ur: Unsigned,
    SoFar: Ord,
    Ul: PrivateCmp<Ur, SoFar>,
{
    type Output = PrivateCmpOut<Ul, Ur, SoFar>;
}

impl<U: Unsigned, B: Bit, SoFar: Ord> PrivateCmp<UInt<U, B>, SoFar> for UTerm {
    type Output = Less;
}

impl<SoFar: Ord> PrivateCmp<UTerm, SoFar> for UTerm {
    type Output = SoFar;
}

type Even<N> = UInt<N, B0>;
type Odd<N> = UInt<N, B1>;

impl<Y> Gcd<Y> for U0 {
    type Output = Y;
}

impl<Xp, Yp> Gcd<Odd<Yp>> for Even<Xp>
where
    Xp: Gcd<Odd<Yp>>,
    Even<Xp>: NonZero,
{
    type Output = Gcf<Xp, Odd<Yp>>;
}
impl<Xp, Yp> Gcd<Odd<Yp>> for Odd<Xp>
where
    Odd<Xp>: Max<Odd<Yp>> + Min<Odd<Yp>>,
    Odd<Yp>: Max<Odd<Xp>>,
    Maximum<Odd<Xp>, Odd<Yp>>: Sub<Minimum<Odd<Xp>, Odd<Yp>>>,
    Diff<Maximum<Odd<Xp>, Odd<Yp>>, Minimum<Odd<Xp>, Odd<Yp>>>: Gcd<Minimum<Odd<Xp>, Odd<Yp>>>,
{
    type Output =
        Gcf<Diff<Maximum<Odd<Xp>, Odd<Yp>>, Minimum<Odd<Xp>, Odd<Yp>>>, Minimum<Odd<Xp>, Odd<Yp>>>;
}

pub trait GetBit<I> {
    type Output;
}

pub type GetBitOut<N, I> = <N as GetBit<I>>::Output;
impl<Un, Bn> GetBit<U0> for UInt<Un, Bn> {
    type Output = Bn;
}
impl<Un, Bn, Ui, Bi> GetBit<UInt<Ui, Bi>> for UInt<Un, Bn>
where
    UInt<Ui, Bi>: Copy + Sub<B1>,
    Un: GetBit<Sub1<UInt<Ui, Bi>>>,
{
    type Output = GetBitOut<Un, Sub1<UInt<Ui, Bi>>>;
}

pub trait SetBit<I, B> {
    type Output;
}
pub type SetBitOut<N, I, B> = <N as SetBit<I, B>>::Output;

impl<N, I, B> SetBit<I, B> for N
where
    N: PrivateSetBit<I, B>,
    PrivateSetBitOut<N, I, B>: Trim,
{
    type Output = TrimOut<PrivateSetBitOut<N, I, B>>;
}

impl<I> PrivateSetBit<I, B1> for UTerm
where
    U1: Shl<I>,
{
    type Output = Shleft<U1, I>;
}

impl<Ul: Unsigned, Bl: Bit, Ur: Unsigned, Br: Bit> Div<UInt<Ur, Br>> for UInt<Ul, Bl>
where
    UInt<Ul, Bl>: Len,
    Length<UInt<Ul, Bl>>: Sub<B1>,
    (): PrivateDiv<UInt<Ul, Bl>, UInt<Ur, Br>, U0, U0, Sub1<Length<UInt<Ul, Bl>>>>,
{
    type Output = PrivateDivQuot<UInt<Ul, Bl>, UInt<Ur, Br>, U0, U0, Sub1<Length<UInt<Ul, Bl>>>>;
    fn div(self, rhs: UInt<Ur, Br>) -> Self::Output {
        loop {}
    }
}

impl<N, D, Q, I> PrivateDiv<N, D, Q, U0, I> for ()
where
    N: GetBit<I>,
    UInt<UTerm, GetBitOut<N, I>>: Trim,
    TrimOut<UInt<UTerm, GetBitOut<N, I>>>: Cmp<D>,
    (): PrivateDivIf<
        N,
        D,
        Q,
        TrimOut<UInt<UTerm, GetBitOut<N, I>>>,
        I,
        Compare<TrimOut<UInt<UTerm, GetBitOut<N, I>>>, D>,
    >,
{
    type Quotient = PrivateDivIfQuot<
        N,
        D,
        Q,
        TrimOut<UInt<UTerm, GetBitOut<N, I>>>,
        I,
        Compare<TrimOut<UInt<UTerm, GetBitOut<N, I>>>, D>,
    >;
    type Remainder = PrivateDivIfRem<
        N,
        D,
        Q,
        TrimOut<UInt<UTerm, GetBitOut<N, I>>>,
        I,
        Compare<TrimOut<UInt<UTerm, GetBitOut<N, I>>>, D>,
    >;
}
impl<N, D, Q, Ur, Br, I> PrivateDiv<N, D, Q, UInt<Ur, Br>, I> for ()
where
    N: GetBit<I>,
    UInt<UInt<Ur, Br>, GetBitOut<N, I>>: Cmp<D>,
    (): PrivateDivIf<
        N,
        D,
        Q,
        UInt<UInt<Ur, Br>, GetBitOut<N, I>>,
        I,
        Compare<UInt<UInt<Ur, Br>, GetBitOut<N, I>>, D>,
    >,
{
    type Quotient = PrivateDivIfQuot<
        N,
        D,
        Q,
        UInt<UInt<Ur, Br>, GetBitOut<N, I>>,
        I,
        Compare<UInt<UInt<Ur, Br>, GetBitOut<N, I>>, D>,
    >;
    type Remainder = PrivateDivIfRem<
        N,
        D,
        Q,
        UInt<UInt<Ur, Br>, GetBitOut<N, I>>,
        I,
        Compare<UInt<UInt<Ur, Br>, GetBitOut<N, I>>, D>,
    >;
}

impl<N, D, Q, R, Ui, Bi> PrivateDivIf<N, D, Q, R, UInt<Ui, Bi>, Less> for ()
where
    UInt<Ui, Bi>: Sub<B1>,
    (): PrivateDiv<N, D, Q, R, Sub1<UInt<Ui, Bi>>>,
{
    type Quotient = PrivateDivQuot<N, D, Q, R, Sub1<UInt<Ui, Bi>>>;
    type Remainder = PrivateDivRem<N, D, Q, R, Sub1<UInt<Ui, Bi>>>;
}
impl<N, D, Q, R, Ui, Bi> PrivateDivIf<N, D, Q, R, UInt<Ui, Bi>, Equal> for ()
where
    UInt<Ui, Bi>: Copy + Sub<B1>,
    Q: SetBit<UInt<Ui, Bi>, B1>,
    (): PrivateDiv<N, D, SetBitOut<Q, UInt<Ui, Bi>, B1>, U0, Sub1<UInt<Ui, Bi>>>,
{
    type Quotient = PrivateDivQuot<N, D, SetBitOut<Q, UInt<Ui, Bi>, B1>, U0, Sub1<UInt<Ui, Bi>>>;
    type Remainder = PrivateDivRem<N, D, SetBitOut<Q, UInt<Ui, Bi>, B1>, U0, Sub1<UInt<Ui, Bi>>>;
}

impl<N, D, Q, R> PrivateDivIf<N, D, Q, R, U0, Less> for () {
    type Quotient = Q;
    type Remainder = R;
}
impl<N, D, Q, R> PrivateDivIf<N, D, Q, R, U0, Equal> for ()
where
    Q: SetBit<U0, B1>,
{
    type Quotient = SetBitOut<Q, U0, B1>;
    type Remainder = U0;
}

impl<U, B, Ur> PrivateMin<Ur, Equal> for UInt<U, B> {
    type Output = UInt<U, B>;
}

impl<U, B, Ur> Min<Ur> for UInt<U, B>
where
    U: Unsigned,
    B: Bit,
    Ur: Unsigned,
    UInt<U, B>: Cmp<Ur> + PrivateMin<Ur, Compare<UInt<U, B>, Ur>>,
{
    type Output = PrivateMinOut<UInt<U, B>, Ur, Compare<UInt<U, B>, Ur>>;
}

impl<U, B, Ur> PrivateMax<Ur, Equal> for UInt<U, B> {
    type Output = UInt<U, B>;
}

impl<U, B, Ur> Max<Ur> for UInt<U, B>
where
    U: Unsigned,
    B: Bit,
    Ur: Unsigned,
    UInt<U, B>: Cmp<Ur> + PrivateMax<Ur, Compare<UInt<U, B>, Ur>>,
{
    type Output = PrivateMaxOut<UInt<U, B>, Ur, Compare<UInt<U, B>, Ur>>;
}

pub struct Greater;

pub struct Less;

pub struct Equal;

impl Ord for Greater {}

impl Ord for Equal {}

use std::ops::{Add, Div, Mul, Rem, Sub};

pub trait EncodingSize {
    type EncodedPolynomialSize;
}
impl<D> EncodingSize for D
where
    D: Mul<U1> + Gcd<U1>,
    Prod<D, U1>: Div<Gcf<D, U1>>,
    Quot<Prod<D, U1>, Gcf<D, U1>>: Div<D>,
{
    type EncodedPolynomialSize = D;
}

pub fn foo<P>()
where
    U2: Mul<P>,
    Prod<U2, P>: Add<U0> + Div<P>,
    <U2 as EncodingSize>::EncodedPolynomialSize: Mul<P>,
    Sum<Prod<U2, P>, U0>: Sub<Prod<U2, P>, Output = U0>,
{
}

fn main() {}
