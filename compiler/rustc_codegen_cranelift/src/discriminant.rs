//! Handling of enum discriminants
//!
//! Adapted from <https://github.com/rust-lang/rust/blob/31c0645b9d2539f47eecb096142474b29dc542f7/compiler/rustc_codegen_ssa/src/mir/place.rs>
//! (<https://github.com/rust-lang/rust/pull/104535>)

use rustc_abi::Primitive::Int;
use rustc_abi::{TagEncoding, Variants};

use crate::prelude::*;

pub(crate) fn codegen_set_discriminant<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    place: CPlace<'tcx>,
    variant_index: VariantIdx,
) {
    let layout = place.layout();
    if layout.for_variant(fx, variant_index).is_uninhabited() {
        return;
    }
    match layout.variants {
        Variants::Empty => unreachable!("we already handled uninhabited types"),
        Variants::Single { index } => {
            assert_eq!(index, variant_index);
        }
        Variants::Multiple {
            tag: _,
            tag_field,
            tag_encoding: TagEncoding::Direct,
            variants: _,
        } => {
            let ptr = place.place_field(fx, tag_field);
            let to = layout.ty.discriminant_for_variant(fx.tcx, variant_index).unwrap().val;
            let to = match ptr.layout().ty.kind() {
                ty::Uint(UintTy::U128) | ty::Int(IntTy::I128) => {
                    let lsb = fx.bcx.ins().iconst(types::I64, to as u64 as i64);
                    let msb = fx.bcx.ins().iconst(types::I64, (to >> 64) as u64 as i64);
                    fx.bcx.ins().iconcat(lsb, msb)
                }
                ty::Uint(_) | ty::Int(_) => {
                    let clif_ty = fx.clif_type(ptr.layout().ty).unwrap();
                    let raw_val = ptr.layout().size.truncate(to);
                    fx.bcx.ins().iconst(clif_ty, raw_val as i64)
                }
                _ => unreachable!(),
            };
            let discr = CValue::by_val(to, ptr.layout());
            ptr.write_cvalue(fx, discr);
        }
        Variants::Multiple {
            tag: _,
            tag_field,
            tag_encoding: TagEncoding::Niche { untagged_variant, ref niche_variants, niche_start },
            variants: _,
        } => {
            if variant_index != untagged_variant {
                let niche = place.place_field(fx, tag_field);
                let niche_type = fx.clif_type(niche.layout().ty).unwrap();
                let niche_value = variant_index.as_u32() - niche_variants.start().as_u32();
                let niche_value = (niche_value as u128).wrapping_add(niche_start);
                let niche_value = match niche_type {
                    types::I128 => {
                        let lsb = fx.bcx.ins().iconst(types::I64, niche_value as u64 as i64);
                        let msb =
                            fx.bcx.ins().iconst(types::I64, (niche_value >> 64) as u64 as i64);
                        fx.bcx.ins().iconcat(lsb, msb)
                    }
                    ty => fx.bcx.ins().iconst(ty, niche_value as i64),
                };
                let niche_llval = CValue::by_val(niche_value, niche.layout());
                niche.write_cvalue(fx, niche_llval);
            }
        }
    }
}

pub(crate) fn codegen_get_discriminant<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    dest: CPlace<'tcx>,
    value: CValue<'tcx>,
    dest_layout: TyAndLayout<'tcx>,
) {
    let layout = value.layout();

    if layout.is_uninhabited() {
        return;
    }

    let (tag_scalar, tag_field, tag_encoding) = match &layout.variants {
        Variants::Empty => unreachable!("we already handled uninhabited types"),
        Variants::Single { index } => {
            let discr_val = layout
                .ty
                .discriminant_for_variant(fx.tcx, *index)
                .map_or(u128::from(index.as_u32()), |discr| discr.val);

            let val = match dest_layout.ty.kind() {
                ty::Uint(UintTy::U128) | ty::Int(IntTy::I128) => {
                    let lsb = fx.bcx.ins().iconst(types::I64, discr_val as u64 as i64);
                    let msb = fx.bcx.ins().iconst(types::I64, (discr_val >> 64) as u64 as i64);
                    fx.bcx.ins().iconcat(lsb, msb)
                }
                ty::Uint(_) | ty::Int(_) => {
                    let clif_ty = fx.clif_type(dest_layout.ty).unwrap();
                    let raw_val = dest_layout.size.truncate(discr_val);
                    fx.bcx.ins().iconst(clif_ty, raw_val as i64)
                }
                _ => unreachable!(),
            };
            let res = CValue::by_val(val, dest_layout);
            dest.write_cvalue(fx, res);
            return;
        }
        Variants::Multiple { tag, tag_field, tag_encoding, variants: _ } => {
            (tag, *tag_field, tag_encoding)
        }
    };

    let cast_to = fx.clif_type(dest_layout.ty).unwrap();

    // Read the tag/niche-encoded discriminant from memory.
    let tag = value.value_field(fx, tag_field);
    let tag = tag.load_scalar(fx);

    // Decode the discriminant (specifically if it's niche-encoded).
    match *tag_encoding {
        TagEncoding::Direct => {
            let signed = match tag_scalar.primitive() {
                Int(_, signed) => signed,
                _ => false,
            };
            let val = clif_intcast(fx, tag, cast_to, signed);
            let res = CValue::by_val(val, dest_layout);
            dest.write_cvalue(fx, res);
        }
        TagEncoding::Niche { untagged_variant, ref niche_variants, niche_start } => {
            let relative_max = niche_variants.end().as_u32() - niche_variants.start().as_u32();

            // We have a subrange `niche_start..=niche_end` inside `range`.
            // If the value of the tag is inside this subrange, it's a
            // "niche value", an increment of the discriminant. Otherwise it
            // indicates the untagged variant.
            // A general algorithm to extract the discriminant from the tag
            // is:
            // relative_tag = tag - niche_start
            // is_niche = relative_tag <= (ule) relative_max
            // discr = if is_niche {
            //     cast(relative_tag) + niche_variants.start()
            // } else {
            //     untagged_variant
            // }
            // However, we will likely be able to emit simpler code.

            let (is_niche, tagged_discr, delta) = if relative_max == 0 {
                // Best case scenario: only one tagged variant. This will
                // likely become just a comparison and a jump.
                // The algorithm is:
                // is_niche = tag == niche_start
                // discr = if is_niche {
                //     niche_start
                // } else {
                //     untagged_variant
                // }
                let is_niche = codegen_icmp_imm(fx, IntCC::Equal, tag, niche_start as i128);
                let tagged_discr =
                    fx.bcx.ins().iconst(cast_to, niche_variants.start().as_u32() as i64);
                (is_niche, tagged_discr, 0)
            } else {
                // The special cases don't apply, so we'll have to go with
                // the general algorithm.
                let niche_start = match fx.bcx.func.dfg.value_type(tag) {
                    types::I128 => {
                        let lsb = fx.bcx.ins().iconst(types::I64, niche_start as u64 as i64);
                        let msb =
                            fx.bcx.ins().iconst(types::I64, (niche_start >> 64) as u64 as i64);
                        fx.bcx.ins().iconcat(lsb, msb)
                    }
                    ty => fx.bcx.ins().iconst(ty, niche_start as i64),
                };
                let relative_discr = fx.bcx.ins().isub(tag, niche_start);
                let cast_tag = clif_intcast(fx, relative_discr, cast_to, false);
                let is_niche = crate::common::codegen_icmp_imm(
                    fx,
                    IntCC::UnsignedLessThanOrEqual,
                    relative_discr,
                    i128::from(relative_max),
                );
                (is_niche, cast_tag, niche_variants.start().as_u32() as u128)
            };

            let tagged_discr = if delta == 0 {
                tagged_discr
            } else {
                let delta = match cast_to {
                    types::I128 => {
                        let lsb = fx.bcx.ins().iconst(types::I64, delta as u64 as i64);
                        let msb = fx.bcx.ins().iconst(types::I64, (delta >> 64) as u64 as i64);
                        fx.bcx.ins().iconcat(lsb, msb)
                    }
                    ty => fx.bcx.ins().iconst(ty, delta as i64),
                };
                fx.bcx.ins().iadd(tagged_discr, delta)
            };

            let untagged_variant = if cast_to == types::I128 {
                let zero = fx.bcx.ins().iconst(types::I64, 0);
                let untagged_variant =
                    fx.bcx.ins().iconst(types::I64, i64::from(untagged_variant.as_u32()));
                fx.bcx.ins().iconcat(untagged_variant, zero)
            } else {
                fx.bcx.ins().iconst(cast_to, i64::from(untagged_variant.as_u32()))
            };
            let discr = fx.bcx.ins().select(is_niche, tagged_discr, untagged_variant);
            let res = CValue::by_val(discr, dest_layout);
            dest.write_cvalue(fx, res);
        }
    }
}
