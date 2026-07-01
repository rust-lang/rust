//@ compile-flags: -Znext-solver
//@ check-pass

// Regression test for trait-system-refactor-initiative#246
// Fixed by eager norm and marking param env as rigid.

pub trait Add<T> {
    type Output;
}

pub trait Sub<T> {
    type Output;
}

pub trait Shl<T> {
    type Output;
}

pub trait Mul<T> {
    type Output;
}

pub trait Div<T> {
    type Output;
}

pub struct B0;
pub struct B1;

type U0 = UTerm;
type U1 = UInt<UTerm, B1>;
type U2 = UInt<UInt<UTerm, B1>, B0>;

type Sum<A, B> = <A as Add<B>>::Output;

type Sub1<A> = <A as Sub<B1>>::Output;

pub type Compare<A, B> = <A as Cmp<B>>::Output;
type Length<T> = <T as Len>::Output;

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
pub struct InvertedUInt<IU, B> {
    _msb: IU,
    _lsb: B,
}

pub trait PrivateSub<Rhs = Self> {
    type Output;
}
pub type PrivateSubOut<A, Rhs> = <A as PrivateSub<Rhs>>::Output;

trait InvertedUnsigned {}
impl InvertedUnsigned for InvertedUTerm {}
impl<IU, B> InvertedUnsigned for InvertedUInt<IU, B> {}

impl<U, B> Invert for UInt<U, B>
where
    U: PrivateInvert<InvertedUInt<InvertedUTerm, B>>,
{
    type Output = PrivateInvertOut<U, InvertedUInt<InvertedUTerm, B>>;
}
impl<IU> PrivateInvert<IU> for UTerm {
    type Output = IU;
}
impl<IU: InvertedUnsigned, U, B> PrivateInvert<IU> for UInt<U, B>
where
    U: PrivateInvert<InvertedUInt<IU, B>>,
{
    type Output = PrivateInvertOut<U, InvertedUInt<IU, B>>;
}
impl Invert for InvertedUTerm {
    type Output = UTerm;
}
impl<IU: InvertedUnsigned, B> Invert for InvertedUInt<IU, B>
where
    IU: PrivateInvert<UInt<UTerm, B>>,
{
    type Output = <IU as PrivateInvert<UInt<UTerm, B>>>::Output;
}
impl<U> PrivateInvert<U> for InvertedUTerm {
    type Output = U;
}
impl<U, IU, B> PrivateInvert<U> for InvertedUInt<IU, B>
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
impl<U> Trim for U
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
}
type PrivateDivQuot<N, D, Q, R, I> = <() as PrivateDiv<N, D, Q, R, I>>::Quotient;
pub trait PrivateDivIf<N, D, Q, R, I, RcmpD> {
    type Quotient;
}
pub type PrivateDivIfQuot<N, D, Q, R, I, RcmpD> =
    <() as PrivateDivIf<N, D, Q, R, I, RcmpD>>::Quotient;

pub trait Cmp<Rhs = Self> {
    type Output;
}

pub trait Len {
    type Output;
}

pub struct UTerm;

pub struct UInt<U, B> {
    _msb: U,
    _lsb: B,
}

impl Len for UTerm {
    type Output = U0;
}
impl<U, B> Len for UInt<U, B>
where
    U: Len,
    Length<U>: Add<B1>,
{
    type Output = <Length<U> as Add<B1>>::Output;
}

impl Add<B1> for UTerm {
    type Output = UInt<UTerm, B1>;
}

impl<U> Add<B1> for UInt<U, B1>
where
    U: Add<B1>,
{
    type Output = UInt<<U as Add<B1>>::Output, B0>;
}

impl Add<UTerm> for UTerm {
    type Output = UTerm;
}
impl<U, B> Add<UInt<U, B>> for UTerm {
    type Output = UInt<U, B>;
}

impl<Ul, Ur> Add<UInt<Ur, B1>> for UInt<Ul, B0>
where
    Ul: Add<Ur>,
{
    type Output = UInt<Sum<Ul, Ur>, B1>;
}

impl Sub<B1> for UInt<UTerm, B1> {
    type Output = UTerm;
}
impl<U> Sub<B1> for UInt<U, B0>
where
    U: Sub<B1>,
{
    type Output = UInt<Sub1<U>, B1>;
}

impl<Ul, Bl, Ur> Sub<Ur> for UInt<Ul, Bl>
where
    UInt<Ul, Bl>: PrivateSub<Ur>,
    PrivateSubOut<UInt<Ul, Bl>, Ur>: Trim,
{
    type Output = TrimOut<PrivateSubOut<UInt<Ul, Bl>, Ur>>;
}
impl<U> PrivateSub<UTerm> for U {
    type Output = U;
}

impl<Ul, Ur> PrivateSub<UInt<Ur, B1>> for UInt<Ul, B1>
where
    Ul: PrivateSub<Ur>,
{
    type Output = UInt<PrivateSubOut<Ul, Ur>, B0>;
}

impl<U, B> Shl<UTerm> for UInt<U, B> {
    type Output = UInt<U, B>;
}

impl<U, B, Ur> Shl<UInt<Ur, B1>> for UInt<U, B>
where
    UInt<Ur, B1>: Sub<B1>,
    UInt<UInt<U, B>, B0>: Shl<Sub1<UInt<Ur, B1>>>,
{
    type Output = <UInt<UInt<U, B>, B0> as Shl<Sub1<UInt<Ur, B1>>>>::Output;
}

impl<U, B> Cmp<UInt<U, B>> for UTerm {
    type Output = Less;
}
impl<Ul, Ur> Cmp<UInt<Ur, B0>> for UInt<Ul, B0>
where
    Ul: PrivateCmp<Ur, Equal>,
{
    type Output = PrivateCmpOut<Ul, Ur, Equal>;
}
impl<Ul, Ur> Cmp<UInt<Ur, B1>> for UInt<Ul, B1>
where
    Ul: PrivateCmp<Ur, Equal>,
{
    type Output = PrivateCmpOut<Ul, Ur, Equal>;
}

impl<Ul, Ur> Cmp<UInt<Ur, B0>> for UInt<Ul, B1>
where
    Ul: PrivateCmp<Ur, Greater>,
{
    type Output = PrivateCmpOut<Ul, Ur, Greater>;
}

impl<Ul, Ur, SoFar> PrivateCmp<UInt<Ur, B1>, SoFar> for UInt<Ul, B1>
where
    Ul: PrivateCmp<Ur, SoFar>,
{
    type Output = PrivateCmpOut<Ul, Ur, SoFar>;
}

impl<U, B, SoFar> PrivateCmp<UInt<U, B>, SoFar> for UTerm {
    type Output = Less;
}

impl<SoFar> PrivateCmp<UTerm, SoFar> for UTerm {
    type Output = SoFar;
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
    UInt<Ui, Bi>: Sub<B1>,
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
    type Output = <U1 as Shl<I>>::Output;
}

impl<Ur, Br> Div<UInt<Ur, Br>> for U2
where
    Length<U2>: Sub<B1>,
    (): PrivateDiv<U2, UInt<Ur, Br>, U0, U0, Sub1<U2>>,
{
    type Output = PrivateDivQuot<U2, UInt<Ur, Br>, U0, U0, U1>;
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
}

impl<N, D, Q, R, Ui, Bi> PrivateDivIf<N, D, Q, R, UInt<Ui, Bi>, Less> for ()
where
    UInt<Ui, Bi>: Sub<B1>,
    (): PrivateDiv<N, D, Q, R, Sub1<UInt<Ui, Bi>>>,
{
    type Quotient = PrivateDivQuot<N, D, Q, R, Sub1<UInt<Ui, Bi>>>;
}
impl<N, D, Q, R, Ui, Bi> PrivateDivIf<N, D, Q, R, UInt<Ui, Bi>, Equal> for ()
where
    UInt<Ui, Bi>: Sub<B1>,
    Q: SetBit<UInt<Ui, Bi>, B1>,
    (): PrivateDiv<N, D, SetBitOut<Q, UInt<Ui, Bi>, B1>, U0, Sub1<UInt<Ui, Bi>>>,
{
    type Quotient = PrivateDivQuot<N, D, SetBitOut<Q, UInt<Ui, Bi>, B1>, U0, Sub1<UInt<Ui, Bi>>>;
}

impl<N, D, Q, R> PrivateDivIf<N, D, Q, R, U0, Less> for () {
    type Quotient = Q;
}
impl<N, D, Q, R> PrivateDivIf<N, D, Q, R, U0, Equal> for ()
where
    Q: SetBit<U0, B1>,
{
    type Quotient = SetBitOut<Q, U0, B1>;
}

pub struct Greater;

pub struct Less;

pub struct Equal;

pub trait EncodingSize {
    type EncodedPolynomialSize;
}
impl EncodingSize for U2
where
    U2: Div<U1>,
    <U2 as Div<U1>>::Output: Div<U2>,
{
    type EncodedPolynomialSize = U2;
}

pub fn foo<P>()
where
    U2: Mul<P, Output: Sub<<U2 as Mul<P>>::Output, Output = U0>>,
    <U2 as EncodingSize>::EncodedPolynomialSize: Mul<P>,
{
}

fn main() {}
