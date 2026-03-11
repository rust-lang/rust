use rustc_hash::FxHashMap;
use rustc_serialize::{Decodable, Encodable};
use rustc_span::{SpanDecoder, SpanEncoder};

use crate::{Interner, PredicateKind, Ty, TyKind};

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

impl<'tcx, I, E> Encodable<E> for Ty<I>
where
    I: Interner,
    E: TyEncoder<'tcx, Interner = I>,
    TyKind<I>: Encodable<E>,
{
    fn encode(&self, e: &mut E) {
        let existing_shorthand = e.type_shorthands().get(self).copied();
        if let Some(shorthand) = existing_shorthand {
            e.emit_usize(shorthand);
            return;
        }

        let kind = crate::inherent::IntoKind::kind(*self);
        let start = e.position();
        kind.encode(e);
        let len = e.position() - start;

        let shorthand = start + SHORTHAND_OFFSET;

        // Get the number of bits that leb128 could fit
        // in the same space as the fully encoded type.
        let leb128_bits = len * 7;

        // Check that the shorthand is a not longer than the
        // full encoding itself, i.e., it's an obvious win.
        if leb128_bits >= 64 || (shorthand as u64) < (1 << leb128_bits) {
            e.type_shorthands().insert(*self, shorthand);
        }
    }
}

impl<'tcx, I, D> Decodable<D> for Ty<I>
where
    I: Interner,
    D: TyDecoder<'tcx, Interner = I>,
    TyKind<I>: Decodable<D>,
{
    fn decode(decoder: &mut D) -> Ty<I> {
        // Handle shorthands first, if we have a usize > 0x80.
        if decoder.positioned_at_shorthand() {
            let pos = decoder.read_usize();
            assert!(pos >= SHORTHAND_OFFSET);
            let shorthand = pos - SHORTHAND_OFFSET;

            decoder.cached_ty_for_shorthand(shorthand, |decoder| {
                decoder.with_position(shorthand, Ty::decode)
            })
        } else {
            let interner = decoder.interner();
            interner.mk_ty_from_kind(TyKind::decode(decoder))
        }
    }
}
