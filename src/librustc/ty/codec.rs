// This module contains some shared code for encoding and decoding various
// things from the `ty` module, and in particular implements support for
// "shorthands" which allow to have pointers back into the already encoded
// stream instead of re-encoding the same thing twice.
//
// The functionality in here is shared between persisting to crate metadata and
// persisting to incr. comp. caches.

use crate::arena::ArenaAllocatable;
use crate::hir::def_id::{DefId, CrateNum};
use crate::infer::canonical::{CanonicalVarInfo, CanonicalVarInfos};
use rustc_data_structures::fx::FxHashMap;
use crate::rustc_serialize::{Decodable, Decoder, Encoder, Encodable, opaque};
use std::hash::Hash;
use std::intrinsics;
use crate::ty::{self, Ty, TyCtxt};
use crate::ty::subst::SubstsRef;
use crate::mir::interpret::Allocation;

/// The shorthand encoding uses an enum's variant index `usize`
/// and is offset by this value so it never matches a real variant.
/// This offset is also chosen so that the first byte is never < 0x80.
pub const SHORTHAND_OFFSET: usize = 0x80;

pub trait EncodableWithShorthand: Clone + Eq + Hash {
    type Variant: Encodable;
    fn variant(&self) -> &Self::Variant;
}

impl<'tcx> EncodableWithShorthand for Ty<'tcx> {
    type Variant = ty::TyKind<'tcx>;
    fn variant(&self) -> &Self::Variant {
        &self.sty
    }
}

impl<'tcx> EncodableWithShorthand for ty::Predicate<'tcx> {
    type Variant = ty::Predicate<'tcx>;
    fn variant(&self) -> &Self::Variant {
        self
    }
}

pub trait TyEncoder: Encoder {
    fn position(&self) -> usize;
}

impl TyEncoder for opaque::Encoder {
    #[inline]
    fn position(&self) -> usize {
        self.position()
    }
}

/// Encode the given value or a previously cached shorthand.
pub fn encode_with_shorthand<E, T, M>(encoder: &mut E,
                                      value: &T,
                                      cache: M)
                                      -> Result<(), E::Error>
    where E: TyEncoder,
          M: for<'b> Fn(&'b mut E) -> &'b mut FxHashMap<T, usize>,
          T: EncodableWithShorthand,
{
    let existing_shorthand = cache(encoder).get(value).cloned();
    if let Some(shorthand) = existing_shorthand {
        return encoder.emit_usize(shorthand);
    }

    let variant = value.variant();

    let start = encoder.position();
    variant.encode(encoder)?;
    let len = encoder.position() - start;

    // The shorthand encoding uses the same usize as the
    // discriminant, with an offset so they can't conflict.
    let discriminant = unsafe { intrinsics::discriminant_value(variant) };
    assert!(discriminant < SHORTHAND_OFFSET as u64);
    let shorthand = start + SHORTHAND_OFFSET;

    // Get the number of bits that leb128 could fit
    // in the same space as the fully encoded type.
    let leb128_bits = len * 7;

    // Check that the shorthand is a not longer than the
    // full encoding itself, i.e., it's an obvious win.
    if leb128_bits >= 64 || (shorthand as u64) < (1 << leb128_bits) {
        cache(encoder).insert(value.clone(), shorthand);
    }

    Ok(())
}

pub fn encode_predicates<'tcx, E, C>(encoder: &mut E,
                                     predicates: &ty::GenericPredicates<'tcx>,
                                     cache: C)
                                     -> Result<(), E::Error>
    where E: TyEncoder,
          C: for<'b> Fn(&'b mut E) -> &'b mut FxHashMap<ty::Predicate<'tcx>, usize>,
{
    predicates.parent.encode(encoder)?;
    predicates.predicates.len().encode(encoder)?;
    for (predicate, span) in &predicates.predicates {
        encode_with_shorthand(encoder, predicate, &cache)?;
        span.encode(encoder)?;
    }
    Ok(())
}

pub trait TyDecoder<'tcx>: Decoder {
    fn tcx(&self) -> TyCtxt<'tcx>;

    fn peek_byte(&self) -> u8;

    fn position(&self) -> usize;

    fn cached_ty_for_shorthand<F>(&mut self,
                                  shorthand: usize,
                                  or_insert_with: F)
                                  -> Result<Ty<'tcx>, Self::Error>
        where F: FnOnce(&mut Self) -> Result<Ty<'tcx>, Self::Error>;

    fn with_position<F, R>(&mut self, pos: usize, f: F) -> R
        where F: FnOnce(&mut Self) -> R;

    fn map_encoded_cnum_to_current(&self, cnum: CrateNum) -> CrateNum;

    fn positioned_at_shorthand(&self) -> bool {
        (self.peek_byte() & (SHORTHAND_OFFSET as u8)) != 0
    }
}

#[inline]
pub fn decode_arena_allocable<D, T: ArenaAllocatable + Decodable>(
    decoder: &mut D,
) -> Result<&'tcx T, D::Error>
where
    D: TyDecoder<'tcx>,
{
    Ok(decoder.tcx().arena.alloc(Decodable::decode(decoder)?))
}

#[inline]
pub fn decode_arena_allocable_slice<D, T: ArenaAllocatable + Decodable>(
    decoder: &mut D,
) -> Result<&'tcx [T], D::Error>
where
    D: TyDecoder<'tcx>,
{
    Ok(decoder.tcx().arena.alloc_from_iter(<Vec<T> as Decodable>::decode(decoder)?))
}

#[inline]
pub fn decode_cnum<D>(decoder: &mut D) -> Result<CrateNum, D::Error>
where
    D: TyDecoder<'tcx>,
{
    let cnum = CrateNum::from_u32(u32::decode(decoder)?);
    Ok(decoder.map_encoded_cnum_to_current(cnum))
}

#[inline]
pub fn decode_ty<D>(decoder: &mut D) -> Result<Ty<'tcx>, D::Error>
where
    D: TyDecoder<'tcx>,
{
    // Handle shorthands first, if we have an usize > 0x80.
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

#[inline]
pub fn decode_predicates<D>(decoder: &mut D) -> Result<ty::GenericPredicates<'tcx>, D::Error>
where
    D: TyDecoder<'tcx>,
{
    Ok(ty::GenericPredicates {
        parent: Decodable::decode(decoder)?,
        predicates: (0..decoder.read_usize()?).map(|_| {
            // Handle shorthands first, if we have an usize > 0x80.
            let predicate = if decoder.positioned_at_shorthand() {
                let pos = decoder.read_usize()?;
                assert!(pos >= SHORTHAND_OFFSET);
                let shorthand = pos - SHORTHAND_OFFSET;

                decoder.with_position(shorthand, ty::Predicate::decode)
            } else {
                ty::Predicate::decode(decoder)
            }?;
            Ok((predicate, Decodable::decode(decoder)?))
        })
        .collect::<Result<Vec<_>, _>>()?,
    })
}

#[inline]
pub fn decode_substs<D>(decoder: &mut D) -> Result<SubstsRef<'tcx>, D::Error>
where
    D: TyDecoder<'tcx>,
{
    let len = decoder.read_usize()?;
    let tcx = decoder.tcx();
    Ok(tcx.mk_substs((0..len).map(|_| Decodable::decode(decoder)))?)
}

#[inline]
pub fn decode_region<D>(decoder: &mut D) -> Result<ty::Region<'tcx>, D::Error>
where
    D: TyDecoder<'tcx>,
{
    Ok(decoder.tcx().mk_region(Decodable::decode(decoder)?))
}

#[inline]
pub fn decode_ty_slice<D>(decoder: &mut D) -> Result<&'tcx ty::List<Ty<'tcx>>, D::Error>
where
    D: TyDecoder<'tcx>,
{
    let len = decoder.read_usize()?;
    Ok(decoder.tcx().mk_type_list((0..len).map(|_| Decodable::decode(decoder)))?)
}

#[inline]
pub fn decode_adt_def<D>(decoder: &mut D) -> Result<&'tcx ty::AdtDef, D::Error>
where
    D: TyDecoder<'tcx>,
{
    let def_id = DefId::decode(decoder)?;
    Ok(decoder.tcx().adt_def(def_id))
}

#[inline]
pub fn decode_existential_predicate_slice<D>(
    decoder: &mut D,
) -> Result<&'tcx ty::List<ty::ExistentialPredicate<'tcx>>, D::Error>
where
    D: TyDecoder<'tcx>,
{
    let len = decoder.read_usize()?;
    Ok(decoder.tcx()
              .mk_existential_predicates((0..len).map(|_| Decodable::decode(decoder)))?)
}

#[inline]
pub fn decode_canonical_var_infos<D>(decoder: &mut D) -> Result<CanonicalVarInfos<'tcx>, D::Error>
where
    D: TyDecoder<'tcx>,
{
    let len = decoder.read_usize()?;
    let interned: Result<Vec<CanonicalVarInfo>, _> = (0..len).map(|_| Decodable::decode(decoder))
                                                             .collect();
    Ok(decoder.tcx()
              .intern_canonical_var_infos(interned?.as_slice()))
}

#[inline]
pub fn decode_const<D>(decoder: &mut D) -> Result<&'tcx ty::Const<'tcx>, D::Error>
where
    D: TyDecoder<'tcx>,
{
    Ok(decoder.tcx().mk_const(Decodable::decode(decoder)?))
}

#[inline]
pub fn decode_allocation<D>(decoder: &mut D) -> Result<&'tcx Allocation, D::Error>
where
    D: TyDecoder<'tcx>,
{
    Ok(decoder.tcx().intern_const_alloc(Decodable::decode(decoder)?))
}

#[macro_export]
macro_rules! __impl_decoder_methods {
    ($($name:ident -> $ty:ty;)*) => {
        $(fn $name(&mut self) -> Result<$ty, Self::Error> {
            self.opaque.$name()
        })*
    }
}

#[macro_export]
macro_rules! impl_arena_allocatable_decoder {
    ([]$args:tt) => {};
    ([decode $(, $attrs:ident)*]
     [[$DecoderName:ident [$($typaram:tt),*]], [$name:ident: $ty:ty], $tcx:lifetime]) => {
        impl<$($typaram),*> SpecializedDecoder<&$tcx $ty> for $DecoderName<$($typaram),*> {
            #[inline]
            fn specialized_decode(&mut self) -> Result<&$tcx $ty, Self::Error> {
                decode_arena_allocable(self)
            }
        }

        impl<$($typaram),*> SpecializedDecoder<&$tcx [$ty]> for $DecoderName<$($typaram),*> {
            #[inline]
            fn specialized_decode(&mut self) -> Result<&$tcx [$ty], Self::Error> {
                decode_arena_allocable_slice(self)
            }
        }
    };
    ([$ignore:ident $(, $attrs:ident)*]$args:tt) => {
        impl_arena_allocatable_decoder!([$($attrs),*]$args);
    };
}

#[macro_export]
macro_rules! impl_arena_allocatable_decoders {
    ($args:tt, [$($a:tt $name:ident: $ty:ty,)*], $tcx:lifetime) => {
        $(
            impl_arena_allocatable_decoder!($a [$args, [$name: $ty], $tcx]);
        )*
    }
}

#[macro_export]
macro_rules! implement_ty_decoder {
    ($DecoderName:ident <$($typaram:tt),*>) => {
        mod __ty_decoder_impl {
            use super::$DecoderName;
            use $crate::infer::canonical::CanonicalVarInfos;
            use $crate::ty;
            use $crate::ty::codec::*;
            use $crate::ty::subst::SubstsRef;
            use $crate::hir::def_id::{CrateNum};
            use crate::rustc_serialize::{Decoder, SpecializedDecoder};
            use std::borrow::Cow;

            impl<$($typaram ),*> Decoder for $DecoderName<$($typaram),*> {
                type Error = String;

                __impl_decoder_methods! {
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

                fn error(&mut self, err: &str) -> Self::Error {
                    self.opaque.error(err)
                }
            }

            // FIXME(#36588) These impls are horribly unsound as they allow
            // the caller to pick any lifetime for 'tcx, including 'static,
            // by using the unspecialized proxies to them.

            arena_types!(impl_arena_allocatable_decoders, [$DecoderName [$($typaram),*]], 'tcx);

            impl<$($typaram),*> SpecializedDecoder<CrateNum>
            for $DecoderName<$($typaram),*> {
                fn specialized_decode(&mut self) -> Result<CrateNum, Self::Error> {
                    decode_cnum(self)
                }
            }

            impl<$($typaram),*> SpecializedDecoder<ty::Ty<'tcx>>
            for $DecoderName<$($typaram),*> {
                fn specialized_decode(&mut self) -> Result<ty::Ty<'tcx>, Self::Error> {
                    decode_ty(self)
                }
            }

            impl<$($typaram),*> SpecializedDecoder<ty::GenericPredicates<'tcx>>
            for $DecoderName<$($typaram),*> {
                fn specialized_decode(&mut self)
                                      -> Result<ty::GenericPredicates<'tcx>, Self::Error> {
                    decode_predicates(self)
                }
            }

            impl<$($typaram),*> SpecializedDecoder<SubstsRef<'tcx>>
            for $DecoderName<$($typaram),*> {
                fn specialized_decode(&mut self) -> Result<SubstsRef<'tcx>, Self::Error> {
                    decode_substs(self)
                }
            }

            impl<$($typaram),*> SpecializedDecoder<ty::Region<'tcx>>
            for $DecoderName<$($typaram),*> {
                fn specialized_decode(&mut self) -> Result<ty::Region<'tcx>, Self::Error> {
                    decode_region(self)
                }
            }

            impl<$($typaram),*> SpecializedDecoder<&'tcx ty::List<ty::Ty<'tcx>>>
            for $DecoderName<$($typaram),*> {
                fn specialized_decode(&mut self)
                                      -> Result<&'tcx ty::List<ty::Ty<'tcx>>, Self::Error> {
                    decode_ty_slice(self)
                }
            }

            impl<$($typaram),*> SpecializedDecoder<&'tcx ty::AdtDef>
            for $DecoderName<$($typaram),*> {
                fn specialized_decode(&mut self) -> Result<&'tcx ty::AdtDef, Self::Error> {
                    decode_adt_def(self)
                }
            }

            impl<$($typaram),*> SpecializedDecoder<&'tcx ty::List<ty::ExistentialPredicate<'tcx>>>
                for $DecoderName<$($typaram),*> {
                fn specialized_decode(&mut self)
                    -> Result<&'tcx ty::List<ty::ExistentialPredicate<'tcx>>, Self::Error> {
                    decode_existential_predicate_slice(self)
                }
            }

            impl<$($typaram),*> SpecializedDecoder<CanonicalVarInfos<'tcx>>
                for $DecoderName<$($typaram),*> {
                fn specialized_decode(&mut self)
                    -> Result<CanonicalVarInfos<'tcx>, Self::Error> {
                    decode_canonical_var_infos(self)
                }
            }

            impl<$($typaram),*> SpecializedDecoder<&'tcx $crate::ty::Const<'tcx>>
            for $DecoderName<$($typaram),*> {
                fn specialized_decode(&mut self) -> Result<&'tcx ty::Const<'tcx>, Self::Error> {
                    decode_const(self)
                }
            }

            impl<$($typaram),*> SpecializedDecoder<&'tcx $crate::mir::interpret::Allocation>
            for $DecoderName<$($typaram),*> {
                fn specialized_decode(
                    &mut self
                ) -> Result<&'tcx $crate::mir::interpret::Allocation, Self::Error> {
                    decode_allocation(self)
                }
            }
        }
    }
}
