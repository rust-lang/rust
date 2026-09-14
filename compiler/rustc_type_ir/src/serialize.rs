use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};

use crate::visit::TypeVisitable;
use crate::{self as ty, Interner, Region, RegionKind, UnsafeBinderInner};

/// A decoder that can reconstruct interned type IR values by supplying the
/// interner that owns the decoded data.
///
/// Some serialized type IR wrappers only store their structural kind. When
/// decoding, those kinds have to be re-interned instead of rebuilt as raw
/// values, so their `Decodable` impls need access to the active interner in
/// addition to the byte stream.
pub trait InternerDecoder: Decoder {
    type Interner: Interner;

    fn interner(&self) -> Self::Interner;
}

// Every binder uses the same wire representation: its ordered declarations,
// followed by its payload. Keep this generic so a closed evidence projection
// can cross MIR and metadata boundaries without losing its region binder.
impl<I: Interner, T: Encodable<E>, E: Encoder> Encodable<E> for ty::Binder<I, T>
where
    I::BoundVarKinds: Encodable<E>,
{
    fn encode(&self, e: &mut E) {
        self.bound_vars().encode(e);
        self.as_ref().skip_binder().encode(e);
    }
}

impl<I: Interner, T: TypeVisitable<I> + Decodable<D>, D: Decoder> Decodable<D> for ty::Binder<I, T>
where
    I::BoundVarKinds: Decodable<D>,
{
    fn decode(decoder: &mut D) -> Self {
        let bound_vars = Decodable::decode(decoder);
        ty::Binder::bind_with_vars(Decodable::decode(decoder), bound_vars)
    }
}

impl<I: Interner, E: Encoder> Encodable<E> for UnsafeBinderInner<I>
where
    I::Ty: Encodable<E>,
    I::BoundVarKinds: Encodable<E>,
{
    fn encode(&self, e: &mut E) {
        self.bound_vars().encode(e);
        self.as_ref().skip_binder().encode(e);
    }
}

impl<I: Interner, D: Decoder> Decodable<D> for UnsafeBinderInner<I>
where
    I::Ty: TypeVisitable<I> + Decodable<D>,
    I::BoundVarKinds: Decodable<D>,
{
    fn decode(decoder: &mut D) -> Self {
        let bound_vars = Decodable::decode(decoder);
        ty::Binder::bind_with_vars(Decodable::decode(decoder), bound_vars).into()
    }
}

impl<I: Interner, E: Encoder> Encodable<E> for Region<I>
where
    RegionKind<I>: Encodable<E>,
{
    fn encode(&self, e: &mut E) {
        self.kind().encode(e);
    }
}

// Decoding a region needs an interner so it can decode a RegionKind and
// re-intern it.
impl<I: Interner, D: InternerDecoder<Interner = I>> Decodable<D> for Region<I>
where
    RegionKind<I>: Decodable<D>,
{
    fn decode(decoder: &mut D) -> Self {
        decoder.interner().intern_region(Decodable::decode(decoder))
    }
}
