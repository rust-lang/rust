//@check-pass

// issue#145741

use std::marker::PhantomData;

pub struct Fn2<A: ?Sized, B: ?Sized, R: ?Sized> {
    _0: PhantomData<fn() -> A>,
    _1: PhantomData<fn() -> B>,
    _2: PhantomData<fn() -> R>,
}

impl<A: ?Sized + Tag, B: ?Sized + Tag, R: ?Sized + Tag> Tag for Fn2<A, B, R> {
    type MaskProjection = MaskToProjection<
        MaskOr<
            ProjectMask<MaskAllOn, <A as Tag>::MaskProjection>,
            MaskOr<
                ProjectMask<MaskAllOn, <B as Tag>::MaskProjection>,
                MaskOr<ProjectMask<MaskAllOn, <R as Tag>::MaskProjection>, Mask>,
            >,
        >,
    >;
}

// Add 1 generic (and it's part of the associated type)
pub struct Fn3<A: ?Sized, B: ?Sized, C: ?Sized, R: ?Sized> {
    _0: PhantomData<fn() -> A>,
    _1: PhantomData<fn() -> B>,
    _2: PhantomData<fn() -> C>,
    _3: PhantomData<fn() -> R>,
}

impl<A: ?Sized + Tag, B: ?Sized + Tag, C: ?Sized + Tag, R: ?Sized + Tag> Tag for Fn3<A, B, C, R> {
    type MaskProjection = MaskToProjection<
        MaskOr<
            ProjectMask<MaskAllOn, <A as Tag>::MaskProjection>,
            MaskOr<
                ProjectMask<MaskAllOn, <B as Tag>::MaskProjection>,
                MaskOr<
                    ProjectMask<MaskAllOn, <C as Tag>::MaskProjection>,
                    MaskOr<ProjectMask<MaskAllOn, <R as Tag>::MaskProjection>, Mask>,
                >,
            >,
        >,
    >;
}

pub struct TypeTuple<_0, _1, _2, _3, _4, _5>(_0, _1, _2, _3, _4, _5);

pub trait TypeTupleAccess {
    type _0;
    type _1;
    type _2;
    type _3;
    type _4;
    type _5;
}

impl<_0, _1, _2, _3, _4, _5> TypeTupleAccess for TypeTuple<_0, _1, _2, _3, _4, _5> {
    type _0 = _0;
    type _1 = _1;
    type _2 = _2;
    type _3 = _3;
    type _4 = _4;
    type _5 = _5;
}

pub trait Projection:
    TypeTupleAccess<
        _0: SlotPicker,
        _1: SlotPicker,
        _2: SlotPicker,
        _3: SlotPicker,
        _4: SlotPicker,
        _5: SlotPicker,
    >
{
}

impl<_0, _1, _2, _3, _4, _5> Projection for TypeTuple<_0, _1, _2, _3, _4, _5>
where
    _0: SlotPicker,
    _1: SlotPicker,
    _2: SlotPicker,
    _3: SlotPicker,
    _4: SlotPicker,
    _5: SlotPicker,
{
}

pub trait Tag: 'static {
    type MaskProjection: Projection;
}

pub trait Tagged {
    type Tag: Tag;
}

pub struct Pick<const I: u8>;

pub trait SlotPicker: 'static {
    type Mask<B: MaskBit>: MaskBits;
}

impl SlotPicker for Pick<0> {
    type Mask<B: MaskBit> = TypeTuple<B, Off, Off, Off, Off, Off>;
}

impl SlotPicker for Pick<1> {
    type Mask<B: MaskBit> = TypeTuple<Off, B, Off, Off, Off, Off>;
}

impl SlotPicker for Pick<2> {
    type Mask<B: MaskBit> = TypeTuple<Off, Off, B, Off, Off, Off>;
}

impl SlotPicker for Pick<3> {
    type Mask<B: MaskBit> = TypeTuple<Off, Off, Off, B, Off, Off>;
}

impl SlotPicker for Pick<4> {
    type Mask<B: MaskBit> = TypeTuple<Off, Off, Off, Off, B, Off>;
}

impl SlotPicker for Pick<5> {
    type Mask<B: MaskBit> = TypeTuple<Off, Off, Off, Off, Off, B>;
}

pub struct Static;

impl SlotPicker for Static {
    type Mask<B: MaskBit> = TypeTuple<Off, Off, Off, Off, Off, Off>;
}

pub type Projector<
  _0 = Static,
  _1 = Static,
  _2 = Static,
  _3 = Static,
  _4 = Static,
  _5 = Static
> =
    TypeTuple<_0, _1, _2, _3, _4, _5>;

pub trait MaskBit {
    type Or<T: MaskBit>: MaskBit;

    type Pick<T: SlotPicker>: SlotPicker;
}

pub type Mask<_0 = Off, _1 = Off, _2 = Off, _3 = Off, _4 = Off, _5 = Off> =
    TypeTuple<_0, _1, _2, _3, _4, _5>;

pub trait MaskBits:
    TypeTupleAccess<_0: MaskBit, _1: MaskBit, _2: MaskBit, _3: MaskBit, _4: MaskBit, _5: MaskBit>
{
}

impl<_0, _1, _2, _3, _4, _5> MaskBits for TypeTuple<_0, _1, _2, _3, _4, _5>
where
    _0: MaskBit,
    _1: MaskBit,
    _2: MaskBit,
    _3: MaskBit,
    _4: MaskBit,
    _5: MaskBit,
{
}

pub struct On;
pub struct Off;

impl MaskBit for On {
    type Or<T: MaskBit> = Self;

    type Pick<T: SlotPicker> = T;
}

impl MaskBit for Off {
    type Or<T: MaskBit> = T;

    type Pick<T: SlotPicker> = Static;
}

pub type MaskAllOn = TypeTuple<On, On, On, On, On, On>;

pub type MaskOr<MaskA, MaskB> = TypeTuple<
    <<MaskA as TypeTupleAccess>::_0 as MaskBit>::Or<<MaskB as TypeTupleAccess>::_0>,
    <<MaskA as TypeTupleAccess>::_1 as MaskBit>::Or<<MaskB as TypeTupleAccess>::_1>,
    <<MaskA as TypeTupleAccess>::_2 as MaskBit>::Or<<MaskB as TypeTupleAccess>::_2>,
    <<MaskA as TypeTupleAccess>::_3 as MaskBit>::Or<<MaskB as TypeTupleAccess>::_3>,
    <<MaskA as TypeTupleAccess>::_4 as MaskBit>::Or<<MaskB as TypeTupleAccess>::_4>,
    <<MaskA as TypeTupleAccess>::_5 as MaskBit>::Or<<MaskB as TypeTupleAccess>::_5>,
>;

pub type ProjectMask<Mask, Projection> = MaskOr<
    MaskOr<
        MaskOr<
            <<Projection as TypeTupleAccess>::_0 as SlotPicker>::Mask<
                <Mask as TypeTupleAccess>::_0,
            >,
            <<Projection as TypeTupleAccess>::_1 as SlotPicker>::Mask<
                <Mask as TypeTupleAccess>::_1,
            >,
        >,
        MaskOr<
            <<Projection as TypeTupleAccess>::_2 as SlotPicker>::Mask<
                <Mask as TypeTupleAccess>::_2,
            >,
            <<Projection as TypeTupleAccess>::_3 as SlotPicker>::Mask<
                <Mask as TypeTupleAccess>::_3,
            >,
        >,
    >,
    MaskOr<
        <<Projection as TypeTupleAccess>::_4 as SlotPicker>::Mask<<Mask as TypeTupleAccess>::_4>,
        <<Projection as TypeTupleAccess>::_5 as SlotPicker>::Mask<<Mask as TypeTupleAccess>::_5>,
    >,
>;

pub type MaskToProjection<Mask> = TypeTuple<
    <<Mask as TypeTupleAccess>::_0 as MaskBit>::Pick<Pick<0>>,
    <<Mask as TypeTupleAccess>::_1 as MaskBit>::Pick<Pick<1>>,
    <<Mask as TypeTupleAccess>::_2 as MaskBit>::Pick<Pick<2>>,
    <<Mask as TypeTupleAccess>::_3 as MaskBit>::Pick<Pick<3>>,
    <<Mask as TypeTupleAccess>::_4 as MaskBit>::Pick<Pick<4>>,
    <<Mask as TypeTupleAccess>::_5 as MaskBit>::Pick<Pick<5>>,
>;

fn main() {}
