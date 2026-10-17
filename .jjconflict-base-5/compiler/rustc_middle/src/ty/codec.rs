//! This module contains some shared code for encoding and decoding various
//! things from the `ty` module, and in particular implements support for
//! "shorthands" which allow to have pointers back into the already encoded
//! stream instead of re-encoding the same thing twice.
//!
//! The functionality in here is shared between persisting to crate metadata and
//! persisting to incr. comp. caches.

use std::hash::Hash;
use std::intrinsics;
use std::marker::DiscriminantKind;

use rustc_data_structures::fx::FxHashMap;
use rustc_serialize::{Decodable, Encodable};
use rustc_span::{SpanDecoder, SpanEncoder};

pub use self::ref_decodable::RefDecodable;
use crate::infer::canonical::{CanonicalVarKind, CanonicalVarKinds};
use crate::mir;
use crate::mir::interpret::{AllocId, ConstAllocation, CtfeProvenance};
use crate::ty::{self, AdtDef, GenericArgsRef, Ty, TyCtxt};

mod ref_decodable;

/// The shorthand encoding uses an enum's variant index `usize`
/// and is offset by this value so it never matches a real variant.
/// This offset is also chosen so that the first byte is never < 0x80.
pub const SHORTHAND_OFFSET: usize = 0x80;

pub trait TyEncoder<'tcx>: SpanEncoder {
    const CLEAR_CROSS_CRATE: bool;

    fn position(&self) -> usize;

    fn type_shorthands(&mut self) -> &mut FxHashMap<Ty<'tcx>, usize>;

    fn predicate_shorthands(&mut self) -> &mut FxHashMap<ty::PredicateKind<'tcx>, usize>;

    fn encode_alloc_id(&mut self, alloc_id: &AllocId);
}

pub trait TyDecoder<'tcx>:
    SpanDecoder + rustc_type_ir::InternerDecoder<Interner = TyCtxt<'tcx>>
{
    const CLEAR_CROSS_CRATE: bool;

    fn cached_ty_for_shorthand<F>(&mut self, shorthand: usize, or_insert_with: F) -> Ty<'tcx>
    where
        F: FnOnce(&mut Self) -> Ty<'tcx>;

    fn with_position<F, R>(&mut self, pos: usize, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R;

    fn positioned_at_shorthand(&self) -> bool {
        (self.peek_byte() & (SHORTHAND_OFFSET as u8)) != 0
    }

    fn decode_alloc_id(&mut self) -> AllocId;
}

pub trait EncodableWithShorthand<'tcx, E: TyEncoder<'tcx>>: Copy + Eq + Hash {
    type Variant: Encodable<E>;
    fn variant(&self) -> &Self::Variant;
}

#[allow(rustc::usage_of_ty_tykind)]
impl<'tcx, E: TyEncoder<'tcx>> EncodableWithShorthand<'tcx, E> for Ty<'tcx> {
    type Variant = ty::TyKind<'tcx>;

    #[inline]
    fn variant(&self) -> &Self::Variant {
        self.kind()
    }
}

impl<'tcx, E: TyEncoder<'tcx>> EncodableWithShorthand<'tcx, E> for ty::PredicateKind<'tcx> {
    type Variant = ty::PredicateKind<'tcx>;

    #[inline]
    fn variant(&self) -> &Self::Variant {
        self
    }
}

/// Encode the given value or a previously cached shorthand.
pub fn encode_with_shorthand<'tcx, E, T, M>(encoder: &mut E, value: &T, cache: M)
where
    E: TyEncoder<'tcx>,
    M: for<'b> Fn(&'b mut E) -> &'b mut FxHashMap<T, usize>,
    T: EncodableWithShorthand<'tcx, E>,
    // The discriminant and shorthand must have the same size.
    T::Variant: DiscriminantKind<Discriminant = isize>,
{
    let existing_shorthand = cache(encoder).get(value).copied();
    if let Some(shorthand) = existing_shorthand {
        encoder.emit_usize(shorthand);
        return;
    }

    let variant = value.variant();

    let start = encoder.position();
    variant.encode(encoder);
    let len = encoder.position() - start;

    // The shorthand encoding uses the same usize as the
    // discriminant, with an offset so they can't conflict.
    let discriminant = intrinsics::discriminant_value(variant);
    assert!(SHORTHAND_OFFSET > discriminant as usize);

    let shorthand = start + SHORTHAND_OFFSET;

    // Get the number of bits that leb128 could fit
    // in the same space as the fully encoded type.
    let leb128_bits = len * 7;

    // Check that the shorthand is a not longer than the
    // full encoding itself, i.e., it's an obvious win.
    if leb128_bits >= 64 || (shorthand as u64) < (1 << leb128_bits) {
        cache(encoder).insert(*value, shorthand);
    }
}

impl<'tcx, E: TyEncoder<'tcx>> Encodable<E> for Ty<'tcx> {
    fn encode(&self, e: &mut E) {
        encode_with_shorthand(e, self, TyEncoder::type_shorthands);
    }
}

impl<'tcx, E: TyEncoder<'tcx>> Encodable<E> for ty::Predicate<'tcx> {
    fn encode(&self, e: &mut E) {
        let kind = self.kind();
        kind.bound_vars().encode(e);
        encode_with_shorthand(e, &kind.skip_binder(), TyEncoder::predicate_shorthands);
    }
}

impl<'tcx, E: TyEncoder<'tcx>> Encodable<E> for ty::Clause<'tcx> {
    fn encode(&self, e: &mut E) {
        self.as_predicate().encode(e);
    }
}

impl<'tcx, E: TyEncoder<'tcx>> Encodable<E> for ty::Pattern<'tcx> {
    fn encode(&self, e: &mut E) {
        self.0.0.encode(e);
    }
}

impl<'tcx, E: TyEncoder<'tcx>> Encodable<E> for ty::ValTree<'tcx> {
    fn encode(&self, e: &mut E) {
        self.0.0.encode(e);
    }
}

impl<'tcx, E: TyEncoder<'tcx>> Encodable<E> for ConstAllocation<'tcx> {
    fn encode(&self, e: &mut E) {
        self.inner().encode(e)
    }
}

impl<'tcx, E: TyEncoder<'tcx>> Encodable<E> for AdtDef<'tcx> {
    fn encode(&self, e: &mut E) {
        self.0.0.encode(e)
    }
}

impl<'tcx, E: TyEncoder<'tcx>> Encodable<E> for AllocId {
    fn encode(&self, e: &mut E) {
        e.encode_alloc_id(self)
    }
}

impl<'tcx, E: TyEncoder<'tcx>> Encodable<E> for CtfeProvenance {
    fn encode(&self, e: &mut E) {
        self.into_parts().encode(e);
    }
}

impl<'tcx, E: TyEncoder<'tcx>> Encodable<E> for ty::ParamEnv<'tcx> {
    fn encode(&self, e: &mut E) {
        self.caller_bounds.encode(e);
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for Ty<'tcx> {
    #[allow(rustc::usage_of_ty_tykind)]
    fn decode(decoder: &mut D) -> Ty<'tcx> {
        // Handle shorthands first, if we have a usize > 0x80.
        if decoder.positioned_at_shorthand() {
            let pos = decoder.read_usize();
            assert!(pos >= SHORTHAND_OFFSET);
            let shorthand = pos - SHORTHAND_OFFSET;

            decoder.cached_ty_for_shorthand(shorthand, |decoder| {
                decoder.with_position(shorthand, Ty::decode)
            })
        } else {
            let tcx = decoder.interner();
            tcx.mk_ty_from_kind(ty::TyKind::decode(decoder))
        }
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for ty::Predicate<'tcx> {
    fn decode(decoder: &mut D) -> ty::Predicate<'tcx> {
        let bound_vars = Decodable::decode(decoder);
        // Handle shorthands first, if we have a usize > 0x80.
        let predicate_kind = ty::Binder::bind_with_vars(
            if decoder.positioned_at_shorthand() {
                let pos = decoder.read_usize();
                assert!(pos >= SHORTHAND_OFFSET);
                let shorthand = pos - SHORTHAND_OFFSET;

                decoder.with_position(shorthand, <ty::PredicateKind<'tcx> as Decodable<D>>::decode)
            } else {
                <ty::PredicateKind<'tcx> as Decodable<D>>::decode(decoder)
            },
            bound_vars,
        );
        decoder.interner().mk_predicate(predicate_kind)
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for ty::Clause<'tcx> {
    fn decode(decoder: &mut D) -> ty::Clause<'tcx> {
        let pred: ty::Predicate<'tcx> = Decodable::decode(decoder);
        pred.expect_clause()
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for GenericArgsRef<'tcx> {
    fn decode(decoder: &mut D) -> Self {
        let len = decoder.read_usize();
        let tcx = decoder.interner();
        tcx.mk_args_from_iter(
            (0..len).map::<ty::GenericArg<'tcx>, _>(|_| Decodable::decode(decoder)),
        )
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for mir::Place<'tcx> {
    fn decode(decoder: &mut D) -> Self {
        let local: mir::Local = Decodable::decode(decoder);
        let len = decoder.read_usize();
        let projection = decoder.interner().mk_place_elems_from_iter(
            (0..len).map::<mir::PlaceElem<'tcx>, _>(|_| Decodable::decode(decoder)),
        );
        mir::Place { local, projection }
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for CanonicalVarKinds<'tcx> {
    fn decode(decoder: &mut D) -> Self {
        let len = decoder.read_usize();
        decoder.interner().mk_canonical_var_infos_from_iter(
            (0..len).map::<CanonicalVarKind<'tcx>, _>(|_| Decodable::decode(decoder)),
        )
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for AllocId {
    fn decode(decoder: &mut D) -> Self {
        decoder.decode_alloc_id()
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for CtfeProvenance {
    fn decode(decoder: &mut D) -> Self {
        let parts = Decodable::decode(decoder);
        CtfeProvenance::from_parts(parts)
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for ty::SymbolName<'tcx> {
    fn decode(decoder: &mut D) -> Self {
        ty::SymbolName::new(decoder.interner(), decoder.read_str())
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for ty::ParamEnv<'tcx> {
    fn decode(d: &mut D) -> Self {
        let caller_bounds = Decodable::decode(d);
        ty::ParamEnv { caller_bounds }
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for ty::Pattern<'tcx> {
    fn decode(decoder: &mut D) -> Self {
        decoder.interner().mk_pat(Decodable::decode(decoder))
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for ty::ValTree<'tcx> {
    fn decode(decoder: &mut D) -> Self {
        decoder.interner().intern_valtree(Decodable::decode(decoder))
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for ConstAllocation<'tcx> {
    fn decode(decoder: &mut D) -> Self {
        decoder.interner().mk_const_alloc(Decodable::decode(decoder))
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for AdtDef<'tcx> {
    fn decode(decoder: &mut D) -> Self {
        decoder.interner().mk_adt_def_from_data(Decodable::decode(decoder))
    }
}

/// Declares implementations of all [`Decoder`](rustc_serialize::Decoder) methods,
/// each of which forwards to a method of the same name on some underlying decoder,
/// typically a field of type [`MemDecoder`](rustc_serialize::opaque::MemDecoder).
///
/// Call this macro within an impl block `impl Decoder for $MyDecoder { ... }`.
pub macro forward_all_decoder_methods_to {
    (
        // Make the caller provide an explicit `self` (using closure syntax),
        // so that `$inner:expr` can refer to `self` without violating hygiene.
        //
        // This isn't an actual closure, because it needs to work for both
        // `&self` and `&mut self` methods.
        |$self:ident| $inner:expr
    ) => {
        #[inline] fn read_usize(&mut $self) -> usize { $inner.read_usize() }
        #[inline] fn read_u128 (&mut $self) -> u128  { $inner.read_u128()  }
        #[inline] fn read_u64  (&mut $self) -> u64   { $inner.read_u64()   }
        #[inline] fn read_u32  (&mut $self) -> u32   { $inner.read_u32()   }
        #[inline] fn read_u16  (&mut $self) -> u16   { $inner.read_u16()   }
        #[inline] fn read_u8   (&mut $self) -> u8    { $inner.read_u8()    }
        #[inline] fn read_isize(&mut $self) -> isize { $inner.read_isize() }
        #[inline] fn read_i128 (&mut $self) -> i128  { $inner.read_i128()  }
        #[inline] fn read_i64  (&mut $self) -> i64   { $inner.read_i64()   }
        #[inline] fn read_i32  (&mut $self) -> i32   { $inner.read_i32()   }
        #[inline] fn read_i16  (&mut $self) -> i16   { $inner.read_i16()   }

        #[inline]
        fn read_raw_bytes(&mut $self, len: usize) -> &[u8] {
            $inner.read_raw_bytes(len)
        }

        #[inline]
        fn peek_byte(&$self) -> u8 {
            $inner.peek_byte()
        }

        #[inline]
        fn position(&$self) -> usize {
            $inner.position()
        }
    }
}
