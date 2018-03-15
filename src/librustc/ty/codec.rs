// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This module contains some shared code for encoding and decoding various
// things from the `ty` module, and in particular implements support for
// "shorthands" which allow to have pointers back into the already encoded
// stream instead of re-encoding the same thing twice.
//
// The functionality in here is shared between persisting to crate metadata and
// persisting to incr. comp. caches.

use hir::def_id::{DefId, CrateNum};
use middle::const_val::ByteArray;
use rustc_data_structures::fx::FxHashMap;
use rustc_serialize::{Decodable, Decoder, Encoder, Encodable};
use std::hash::Hash;
use std::intrinsics;
use ty::{self, Ty, TyCtxt};
use ty::subst::Substs;

/// The shorthand encoding uses an enum's variant index `usize`
/// and is offset by this value so it never matches a real variant.
/// This offset is also chosen so that the first byte is never < 0x80.
pub const SHORTHAND_OFFSET: usize = 0x80;

pub trait EncodableWithShorthand: Clone + Eq + Hash {
    type Variant: Encodable;
    fn variant(&self) -> &Self::Variant;
}

impl<'tcx> EncodableWithShorthand for Ty<'tcx> {
    type Variant = ty::TypeVariants<'tcx>;
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
    // full encoding itself, i.e. it's an obvious win.
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
    for predicate in &predicates.predicates {
        encode_with_shorthand(encoder, predicate, &cache)?
    }
    Ok(())
}

pub trait TyDecoder<'a, 'tcx: 'a>: Decoder {

    fn tcx(&self) -> TyCtxt<'a, 'tcx, 'tcx>;

    fn peek_byte(&self) -> u8;

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

pub fn decode_cnum<'a, 'tcx, D>(decoder: &mut D) -> Result<CrateNum, D::Error>
    where D: TyDecoder<'a, 'tcx>,
          'tcx: 'a,
{
    let cnum = CrateNum::from_u32(u32::decode(decoder)?);
    Ok(decoder.map_encoded_cnum_to_current(cnum))
}

pub fn decode_ty<'a, 'tcx, D>(decoder: &mut D) -> Result<Ty<'tcx>, D::Error>
    where D: TyDecoder<'a, 'tcx>,
          'tcx: 'a,
{
    // Handle shorthands first, if we have an usize > 0x80.
    // if self.opaque.data[self.opaque.position()] & 0x80 != 0 {
    if decoder.positioned_at_shorthand() {
        let pos = decoder.read_usize()?;
        assert!(pos >= SHORTHAND_OFFSET);
        let shorthand = pos - SHORTHAND_OFFSET;

        decoder.cached_ty_for_shorthand(shorthand, |decoder| {
            decoder.with_position(shorthand, Ty::decode)
        })
    } else {
        let tcx = decoder.tcx();
        Ok(tcx.mk_ty(ty::TypeVariants::decode(decoder)?))
    }
}

pub fn decode_predicates<'a, 'tcx, D>(decoder: &mut D)
                                      -> Result<ty::GenericPredicates<'tcx>, D::Error>
    where D: TyDecoder<'a, 'tcx>,
          'tcx: 'a,
{
    Ok(ty::GenericPredicates {
        parent: Decodable::decode(decoder)?,
        predicates: (0..decoder.read_usize()?).map(|_| {
                // Handle shorthands first, if we have an usize > 0x80.
                if decoder.positioned_at_shorthand() {
                    let pos = decoder.read_usize()?;
                    assert!(pos >= SHORTHAND_OFFSET);
                    let shorthand = pos - SHORTHAND_OFFSET;

                    decoder.with_position(shorthand, ty::Predicate::decode)
                } else {
                    ty::Predicate::decode(decoder)
                }
            })
            .collect::<Result<Vec<_>, _>>()?,
    })
}

pub fn decode_substs<'a, 'tcx, D>(decoder: &mut D) -> Result<&'tcx Substs<'tcx>, D::Error>
    where D: TyDecoder<'a, 'tcx>,
          'tcx: 'a,
{
    let len = decoder.read_usize()?;
    let tcx = decoder.tcx();
    Ok(tcx.mk_substs((0..len).map(|_| Decodable::decode(decoder)))?)
}

pub fn decode_region<'a, 'tcx, D>(decoder: &mut D) -> Result<ty::Region<'tcx>, D::Error>
    where D: TyDecoder<'a, 'tcx>,
          'tcx: 'a,
{
    Ok(decoder.tcx().mk_region(Decodable::decode(decoder)?))
}

pub fn decode_ty_slice<'a, 'tcx, D>(decoder: &mut D)
                                    -> Result<&'tcx ty::Slice<Ty<'tcx>>, D::Error>
    where D: TyDecoder<'a, 'tcx>,
          'tcx: 'a,
{
    let len = decoder.read_usize()?;
    Ok(decoder.tcx().mk_type_list((0..len).map(|_| Decodable::decode(decoder)))?)
}

pub fn decode_adt_def<'a, 'tcx, D>(decoder: &mut D)
                                   -> Result<&'tcx ty::AdtDef, D::Error>
    where D: TyDecoder<'a, 'tcx>,
          'tcx: 'a,
{
    let def_id = DefId::decode(decoder)?;
    Ok(decoder.tcx().adt_def(def_id))
}

pub fn decode_existential_predicate_slice<'a, 'tcx, D>(decoder: &mut D)
    -> Result<&'tcx ty::Slice<ty::ExistentialPredicate<'tcx>>, D::Error>
    where D: TyDecoder<'a, 'tcx>,
          'tcx: 'a,
{
    let len = decoder.read_usize()?;
    Ok(decoder.tcx()
              .mk_existential_predicates((0..len).map(|_| Decodable::decode(decoder)))?)
}

pub fn decode_byte_array<'a, 'tcx, D>(decoder: &mut D)
                                      -> Result<ByteArray<'tcx>, D::Error>
    where D: TyDecoder<'a, 'tcx>,
          'tcx: 'a,
{
    Ok(ByteArray {
        data: decoder.tcx().alloc_byte_array(&Vec::decode(decoder)?)
    })
}

pub fn decode_const<'a, 'tcx, D>(decoder: &mut D)
                                 -> Result<&'tcx ty::Const<'tcx>, D::Error>
    where D: TyDecoder<'a, 'tcx>,
          'tcx: 'a,
{
    Ok(decoder.tcx().mk_const(Decodable::decode(decoder)?))
}
