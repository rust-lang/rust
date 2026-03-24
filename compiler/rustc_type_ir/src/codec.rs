use rustc_hash::FxHashMap;
use rustc_serialize::{Decodable, Encodable};
use rustc_span::{SpanDecoder, SpanEncoder};

use crate::{Interner, PredicateKind, Ty};

/// The shorthand encoding uses an enum's variant index `usize`
/// and is offset by this value so it never matches a real variant.
/// This offset is also chosen so that the first byte is never < 0x80.
pub const SHORTHAND_OFFSET: usize = 0x80;

pub trait TyEncoder<'tcx>: SpanEncoder {
    type Interner: Interner;

    const CLEAR_CROSS_CRATE: bool;

    fn position(&self) -> usize;

    fn type_shorthands(&mut self) -> &mut FxHashMap<Ty<Self::Interner>, usize>;

    fn predicate_shorthands(&mut self) -> &mut FxHashMap<PredicateKind<Self::Interner>, usize>;

    fn encode_alloc_id(&mut self, alloc_id: &<Self::Interner as Interner>::AllocId);
}

pub trait TyDecoder<'tcx>: SpanDecoder {
    type Interner: Interner;

    const CLEAR_CROSS_CRATE: bool;

    fn interner(&self) -> Self::Interner;

    fn cached_ty_for_shorthand<F>(
        &mut self,
        shorthand: usize,
        or_insert_with: F,
    ) -> Ty<Self::Interner>
    where
        F: FnOnce(&mut Self) -> Ty<Self::Interner>;

    fn with_position<F, R>(&mut self, pos: usize, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R;

    fn positioned_at_shorthand(&self) -> bool {
        (self.peek_byte() & (SHORTHAND_OFFSET as u8)) != 0
    }

    fn decode_alloc_id(&mut self) -> <Self::Interner as Interner>::AllocId;
}

pub trait TyCodec<'tcx>: Interner {
    fn encode_ty<E>(ty: Ty<Self>, e: &mut E)
    where
        E: TyEncoder<'tcx, Interner = Self>;

    fn decode_ty<D>(decoder: &mut D) -> Ty<Self>
    where
        D: TyDecoder<'tcx, Interner = Self>;
}

impl<'tcx, I, E> Encodable<E> for Ty<I>
where
    I: TyCodec<'tcx>,
    E: TyEncoder<'tcx, Interner = I>,
{
    fn encode(&self, e: &mut E) {
        I::encode_ty(*self, e);
    }
}

impl<'tcx, I, D> Decodable<D> for Ty<I>
where
    I: TyCodec<'tcx>,
    D: TyDecoder<'tcx, Interner = I>,
{
    fn decode(decoder: &mut D) -> Ty<I> {
        I::decode_ty(decoder)
    }
}
