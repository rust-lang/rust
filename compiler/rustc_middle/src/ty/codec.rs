// This module contains some shared code for encoding and decoding various
// things from the `ty` module, and in particular implements support for
// "shorthands" which allow to have pointers back into the already encoded
// stream instead of re-encoding the same thing twice.
//
// The functionality in here is shared between persisting to crate metadata and
// persisting to incr. comp. caches.

use crate::arena::ArenaAllocatable;
use crate::infer::canonical::{CanonicalVarInfo, CanonicalVarInfos};
use crate::mir::{
    self,
    interpret::{AllocId, Allocation},
};
use crate::thir;
use crate::ty::subst::SubstsRef;
use crate::ty::{self, List, Ty, TyCtxt};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefId;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use rustc_span::Span;
use std::hash::Hash;
use std::intrinsics;
use std::marker::DiscriminantKind;

/// The shorthand encoding uses an enum's variant index `usize`
/// and is offset by this value so it never matches a real variant.
/// This offset is also chosen so that the first byte is never < 0x80.
pub const SHORTHAND_OFFSET: usize = 0x80;

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

pub trait TyEncoder<'tcx>: Encoder {
    const CLEAR_CROSS_CRATE: bool;

    fn position(&self) -> usize;
    fn type_shorthands(&mut self) -> &mut FxHashMap<Ty<'tcx>, usize>;
    fn predicate_shorthands(&mut self) -> &mut FxHashMap<ty::PredicateKind<'tcx>, usize>;
    fn encode_alloc_id(&mut self, alloc_id: &AllocId) -> Result<(), Self::Error>;
}

/// Trait for decoding to a reference.
///
/// This is a separate trait from `Decodable` so that we can implement it for
/// upstream types, such as `FxHashSet`.
///
/// The `TyDecodable` derive macro will use this trait for fields that are
/// references (and don't use a type alias to hide that).
///
/// `Decodable` can still be implemented in cases where `Decodable` is required
/// by a trait bound.
pub trait RefDecodable<'tcx, D: TyDecoder<'tcx>> {
    fn decode(d: &mut D) -> Result<&'tcx Self, D::Error>;
}

/// Encode the given value or a previously cached shorthand.
pub fn encode_with_shorthand<E, T, M>(encoder: &mut E, value: &T, cache: M) -> Result<(), E::Error>
where
    E: TyEncoder<'tcx>,
    M: for<'b> Fn(&'b mut E) -> &'b mut FxHashMap<T, usize>,
    T: EncodableWithShorthand<'tcx, E>,
    // The discriminant and shorthand must have the same size.
    T::Variant: DiscriminantKind<Discriminant = isize>,
{
    let existing_shorthand = cache(encoder).get(value).copied();
    if let Some(shorthand) = existing_shorthand {
        return encoder.emit_usize(shorthand);
    }

    let variant = value.variant();

    let start = encoder.position();
    variant.encode(encoder)?;
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

    Ok(())
}

impl<'tcx, E: TyEncoder<'tcx>> Encodable<E> for Ty<'tcx> {
    fn encode(&self, e: &mut E) -> Result<(), E::Error> {
        encode_with_shorthand(e, self, TyEncoder::type_shorthands)
    }
}

impl<'tcx, E: TyEncoder<'tcx>> Encodable<E> for ty::Binder<'tcx, ty::PredicateKind<'tcx>> {
    fn encode(&self, e: &mut E) -> Result<(), E::Error> {
        self.bound_vars().encode(e)?;
        encode_with_shorthand(e, &self.skip_binder(), TyEncoder::predicate_shorthands)
    }
}

impl<'tcx, E: TyEncoder<'tcx>> Encodable<E> for ty::Predicate<'tcx> {
    fn encode(&self, e: &mut E) -> Result<(), E::Error> {
        self.kind().encode(e)
    }
}

impl<'tcx, E: TyEncoder<'tcx>> Encodable<E> for AllocId {
    fn encode(&self, e: &mut E) -> Result<(), E::Error> {
        e.encode_alloc_id(self)
    }
}

macro_rules! encodable_via_deref {
    ($($t:ty),+) => {
        $(impl<'tcx, E: TyEncoder<'tcx>> Encodable<E> for $t {
            fn encode(&self, e: &mut E) -> Result<(), E::Error> {
                (**self).encode(e)
            }
        })*
    }
}

encodable_via_deref! {
    &'tcx ty::TypeckResults<'tcx>,
    ty::Region<'tcx>,
    &'tcx mir::Body<'tcx>,
    &'tcx mir::UnsafetyCheckResult,
    &'tcx mir::BorrowCheckResult<'tcx>,
    &'tcx mir::coverage::CodeRegion
}

pub trait TyDecoder<'tcx>: Decoder {
    const CLEAR_CROSS_CRATE: bool;

    fn tcx(&self) -> TyCtxt<'tcx>;

    fn peek_byte(&self) -> u8;

    fn position(&self) -> usize;

    fn cached_ty_for_shorthand<F>(
        &mut self,
        shorthand: usize,
        or_insert_with: F,
    ) -> Result<Ty<'tcx>, Self::Error>
    where
        F: FnOnce(&mut Self) -> Result<Ty<'tcx>, Self::Error>;

    fn with_position<F, R>(&mut self, pos: usize, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R;

    fn positioned_at_shorthand(&self) -> bool {
        (self.peek_byte() & (SHORTHAND_OFFSET as u8)) != 0
    }

    fn decode_alloc_id(&mut self) -> Result<AllocId, Self::Error>;
}

#[inline]
fn decode_arena_allocable<'tcx, D, T: ArenaAllocatable<'tcx> + Decodable<D>>(
    decoder: &mut D,
) -> Result<&'tcx T, D::Error>
where
    D: TyDecoder<'tcx>,
{
    Ok(decoder.tcx().arena.alloc(Decodable::decode(decoder)?))
}

#[inline]
fn decode_arena_allocable_slice<'tcx, D, T: ArenaAllocatable<'tcx> + Decodable<D>>(
    decoder: &mut D,
) -> Result<&'tcx [T], D::Error>
where
    D: TyDecoder<'tcx>,
{
    Ok(decoder.tcx().arena.alloc_from_iter(<Vec<T> as Decodable<D>>::decode(decoder)?))
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for Ty<'tcx> {
    #[allow(rustc::usage_of_ty_tykind)]
    fn decode(decoder: &mut D) -> Result<Ty<'tcx>, D::Error> {
        // Handle shorthands first, if we have a usize > 0x80.
        if decoder.positioned_at_shorthand() {
            let pos = decoder.read_usize()?;
            assert!(pos >= SHORTHAND_OFFSET);
            let shorthand = pos - SHORTHAND_OFFSET;

            decoder.cached_ty_for_shorthand(shorthand, |decoder| {
                decoder.with_position(shorthand, Ty::decode)
            })
        } else {
            let tcx = decoder.tcx();
            Ok(tcx.mk_ty(ty::TyKind::decode(decoder)?))
        }
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for ty::Binder<'tcx, ty::PredicateKind<'tcx>> {
    fn decode(decoder: &mut D) -> Result<ty::Binder<'tcx, ty::PredicateKind<'tcx>>, D::Error> {
        let bound_vars = Decodable::decode(decoder)?;
        // Handle shorthands first, if we have a usize > 0x80.
        Ok(ty::Binder::bind_with_vars(
            if decoder.positioned_at_shorthand() {
                let pos = decoder.read_usize()?;
                assert!(pos >= SHORTHAND_OFFSET);
                let shorthand = pos - SHORTHAND_OFFSET;

                decoder.with_position(shorthand, ty::PredicateKind::decode)?
            } else {
                ty::PredicateKind::decode(decoder)?
            },
            bound_vars,
        ))
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for ty::Predicate<'tcx> {
    fn decode(decoder: &mut D) -> Result<ty::Predicate<'tcx>, D::Error> {
        let predicate_kind = Decodable::decode(decoder)?;
        let predicate = decoder.tcx().mk_predicate(predicate_kind);
        Ok(predicate)
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for SubstsRef<'tcx> {
    fn decode(decoder: &mut D) -> Result<Self, D::Error> {
        let len = decoder.read_usize()?;
        let tcx = decoder.tcx();
        tcx.mk_substs((0..len).map(|_| Decodable::decode(decoder)))
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for mir::Place<'tcx> {
    fn decode(decoder: &mut D) -> Result<Self, D::Error> {
        let local: mir::Local = Decodable::decode(decoder)?;
        let len = decoder.read_usize()?;
        let projection: &'tcx List<mir::PlaceElem<'tcx>> =
            decoder.tcx().mk_place_elems((0..len).map(|_| Decodable::decode(decoder)))?;
        Ok(mir::Place { local, projection })
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for ty::Region<'tcx> {
    fn decode(decoder: &mut D) -> Result<Self, D::Error> {
        Ok(decoder.tcx().mk_region(Decodable::decode(decoder)?))
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for CanonicalVarInfos<'tcx> {
    fn decode(decoder: &mut D) -> Result<Self, D::Error> {
        let len = decoder.read_usize()?;
        let interned: Result<Vec<CanonicalVarInfo<'tcx>>, _> =
            (0..len).map(|_| Decodable::decode(decoder)).collect();
        Ok(decoder.tcx().intern_canonical_var_infos(interned?.as_slice()))
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for AllocId {
    fn decode(decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_alloc_id()
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for ty::SymbolName<'tcx> {
    fn decode(decoder: &mut D) -> Result<Self, D::Error> {
        Ok(ty::SymbolName::new(decoder.tcx(), &decoder.read_str()?))
    }
}

macro_rules! impl_decodable_via_ref {
    ($($t:ty),+) => {
        $(impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for $t {
            fn decode(decoder: &mut D) -> Result<Self, D::Error> {
                RefDecodable::decode(decoder)
            }
        })*
    }
}

impl<'tcx, D: TyDecoder<'tcx>> RefDecodable<'tcx, D> for ty::AdtDef {
    fn decode(decoder: &mut D) -> Result<&'tcx Self, D::Error> {
        let def_id = <DefId as Decodable<D>>::decode(decoder)?;
        Ok(decoder.tcx().adt_def(def_id))
    }
}

impl<'tcx, D: TyDecoder<'tcx>> RefDecodable<'tcx, D> for ty::List<Ty<'tcx>> {
    fn decode(decoder: &mut D) -> Result<&'tcx Self, D::Error> {
        let len = decoder.read_usize()?;
        decoder.tcx().mk_type_list((0..len).map(|_| Decodable::decode(decoder)))
    }
}

impl<'tcx, D: TyDecoder<'tcx>> RefDecodable<'tcx, D>
    for ty::List<ty::Binder<'tcx, ty::ExistentialPredicate<'tcx>>>
{
    fn decode(decoder: &mut D) -> Result<&'tcx Self, D::Error> {
        let len = decoder.read_usize()?;
        decoder.tcx().mk_poly_existential_predicates((0..len).map(|_| Decodable::decode(decoder)))
    }
}

impl<'tcx, D: TyDecoder<'tcx>> RefDecodable<'tcx, D> for ty::Const<'tcx> {
    fn decode(decoder: &mut D) -> Result<&'tcx Self, D::Error> {
        Ok(decoder.tcx().mk_const(Decodable::decode(decoder)?))
    }
}

impl<'tcx, D: TyDecoder<'tcx>> RefDecodable<'tcx, D> for [ty::ValTree<'tcx>] {
    fn decode(decoder: &mut D) -> Result<&'tcx Self, D::Error> {
        Ok(decoder.tcx().arena.alloc_from_iter(
            (0..decoder.read_usize()?)
                .map(|_| Decodable::decode(decoder))
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

impl<'tcx, D: TyDecoder<'tcx>> RefDecodable<'tcx, D> for Allocation {
    fn decode(decoder: &mut D) -> Result<&'tcx Self, D::Error> {
        Ok(decoder.tcx().intern_const_alloc(Decodable::decode(decoder)?))
    }
}

impl<'tcx, D: TyDecoder<'tcx>> RefDecodable<'tcx, D> for [(ty::Predicate<'tcx>, Span)] {
    fn decode(decoder: &mut D) -> Result<&'tcx Self, D::Error> {
        Ok(decoder.tcx().arena.alloc_from_iter(
            (0..decoder.read_usize()?)
                .map(|_| Decodable::decode(decoder))
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

impl<'tcx, D: TyDecoder<'tcx>> RefDecodable<'tcx, D> for [thir::abstract_const::Node<'tcx>] {
    fn decode(decoder: &mut D) -> Result<&'tcx Self, D::Error> {
        Ok(decoder.tcx().arena.alloc_from_iter(
            (0..decoder.read_usize()?)
                .map(|_| Decodable::decode(decoder))
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

impl<'tcx, D: TyDecoder<'tcx>> RefDecodable<'tcx, D> for [thir::abstract_const::NodeId] {
    fn decode(decoder: &mut D) -> Result<&'tcx Self, D::Error> {
        Ok(decoder.tcx().arena.alloc_from_iter(
            (0..decoder.read_usize()?)
                .map(|_| Decodable::decode(decoder))
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

impl<'tcx, D: TyDecoder<'tcx>> RefDecodable<'tcx, D> for ty::List<ty::BoundVariableKind> {
    fn decode(decoder: &mut D) -> Result<&'tcx Self, D::Error> {
        let len = decoder.read_usize()?;
        decoder.tcx().mk_bound_variable_kinds((0..len).map(|_| Decodable::decode(decoder)))
    }
}

impl_decodable_via_ref! {
    &'tcx ty::TypeckResults<'tcx>,
    &'tcx ty::List<Ty<'tcx>>,
    &'tcx ty::List<ty::Binder<'tcx, ty::ExistentialPredicate<'tcx>>>,
    &'tcx Allocation,
    &'tcx mir::Body<'tcx>,
    &'tcx mir::UnsafetyCheckResult,
    &'tcx mir::BorrowCheckResult<'tcx>,
    &'tcx mir::coverage::CodeRegion,
    &'tcx ty::List<ty::BoundVariableKind>
}

#[macro_export]
macro_rules! __impl_decoder_methods {
    ($($name:ident -> $ty:ty;)*) => {
        $(
            #[inline]
            fn $name(&mut self) -> Result<$ty, Self::Error> {
                self.opaque.$name()
            }
        )*
    }
}

macro_rules! impl_arena_allocatable_decoder {
    ([]$args:tt) => {};
    ([decode $(, $attrs:ident)*]
     [[$name:ident: $ty:ty], $tcx:lifetime]) => {
        impl<$tcx, D: TyDecoder<$tcx>> RefDecodable<$tcx, D> for $ty {
            #[inline]
            fn decode(decoder: &mut D) -> Result<&$tcx Self, D::Error> {
                decode_arena_allocable(decoder)
            }
        }

        impl<$tcx, D: TyDecoder<$tcx>> RefDecodable<$tcx, D> for [$ty] {
            #[inline]
            fn decode(decoder: &mut D) -> Result<&$tcx Self, D::Error> {
                decode_arena_allocable_slice(decoder)
            }
        }
    };
    ([$ignore:ident $(, $attrs:ident)*]$args:tt) => {
        impl_arena_allocatable_decoder!([$($attrs),*]$args);
    };
}

macro_rules! impl_arena_allocatable_decoders {
    ([$($a:tt $name:ident: $ty:ty,)*], $tcx:lifetime) => {
        $(
            impl_arena_allocatable_decoder!($a [[$name: $ty], $tcx]);
        )*
    }
}

rustc_hir::arena_types!(impl_arena_allocatable_decoders, 'tcx);
arena_types!(impl_arena_allocatable_decoders, 'tcx);

#[macro_export]
macro_rules! implement_ty_decoder {
    ($DecoderName:ident <$($typaram:tt),*>) => {
        mod __ty_decoder_impl {
            use std::borrow::Cow;
            use rustc_serialize::Decoder;

            use super::$DecoderName;

            impl<$($typaram ),*> Decoder for $DecoderName<$($typaram),*> {
                type Error = String;

                $crate::__impl_decoder_methods! {
                    read_nil -> ();

                    read_u128 -> u128;
                    read_u64 -> u64;
                    read_u32 -> u32;
                    read_u16 -> u16;
                    read_u8 -> u8;
                    read_usize -> usize;

                    read_i128 -> i128;
                    read_i64 -> i64;
                    read_i32 -> i32;
                    read_i16 -> i16;
                    read_i8 -> i8;
                    read_isize -> isize;

                    read_bool -> bool;
                    read_f64 -> f64;
                    read_f32 -> f32;
                    read_char -> char;
                    read_str -> Cow<'_, str>;
                }

                #[inline]
                fn read_raw_bytes_into(&mut self, bytes: &mut [u8]) -> Result<(), Self::Error> {
                    self.opaque.read_raw_bytes_into(bytes)
                }

                fn error(&mut self, err: &str) -> Self::Error {
                    self.opaque.error(err)
                }
            }
        }
    }
}

macro_rules! impl_binder_encode_decode {
    ($($t:ty),+ $(,)?) => {
        $(
            impl<'tcx, E: TyEncoder<'tcx>> Encodable<E> for ty::Binder<'tcx, $t> {
                fn encode(&self, e: &mut E) -> Result<(), E::Error> {
                    self.bound_vars().encode(e)?;
                    self.as_ref().skip_binder().encode(e)
                }
            }
            impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for ty::Binder<'tcx, $t> {
                fn decode(decoder: &mut D) -> Result<Self, D::Error> {
                    let bound_vars = Decodable::decode(decoder)?;
                    Ok(ty::Binder::bind_with_vars(Decodable::decode(decoder)?, bound_vars))
                }
            }
        )*
    }
}

impl_binder_encode_decode! {
    &'tcx ty::List<Ty<'tcx>>,
    ty::FnSig<'tcx>,
    ty::ExistentialPredicate<'tcx>,
    ty::TraitRef<'tcx>,
    Vec<ty::GeneratorInteriorTypeCause<'tcx>>,
}
