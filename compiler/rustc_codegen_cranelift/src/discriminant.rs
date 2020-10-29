//! Handling of enum discriminants
//!
//! Adapted from https://github.com/rust-lang/rust/blob/d760df5aea483aae041c9a241e7acacf48f75035/src/librustc_codegen_ssa/mir/place.rs

use rustc_target::abi::{Int, TagEncoding, Variants};

use crate::prelude::*;

pub(crate) fn codegen_set_discriminant<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    place: CPlace<'tcx>,
    variant_index: VariantIdx,
) {
    let layout = place.layout();
    if layout.for_variant(fx, variant_index).abi.is_uninhabited() {
        return;
    }
    match layout.variants {
        Variants::Single { index } => {
            assert_eq!(index, variant_index);
        }
        Variants::Multiple {
            tag: _,
            tag_field,
            tag_encoding: TagEncoding::Direct,
            variants: _,
        } => {
            let ptr = place.place_field(fx, mir::Field::new(tag_field));
            let to = layout
                .ty
                .discriminant_for_variant(fx.tcx, variant_index)
                .unwrap()
                .val
                .into();
            let discr = CValue::const_val(fx, ptr.layout(), to);
            ptr.write_cvalue(fx, discr);
        }
        Variants::Multiple {
            tag: _,
            tag_field,
            tag_encoding:
                TagEncoding::Niche {
                    dataful_variant,
                    ref niche_variants,
                    niche_start,
                },
            variants: _,
        } => {
            if variant_index != dataful_variant {
                let niche = place.place_field(fx, mir::Field::new(tag_field));
                let niche_value = variant_index.as_u32() - niche_variants.start().as_u32();
                let niche_value = u128::from(niche_value).wrapping_add(niche_start);
                let niche_llval = CValue::const_val(fx, niche.layout(), niche_value.into());
                niche.write_cvalue(fx, niche_llval);
            }
        }
    }
}

pub(crate) fn codegen_get_discriminant<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    value: CValue<'tcx>,
    dest_layout: TyAndLayout<'tcx>,
) -> CValue<'tcx> {
    let layout = value.layout();

    if layout.abi == Abi::Uninhabited {
        return trap_unreachable_ret_value(
            fx,
            dest_layout,
            "[panic] Tried to get discriminant for uninhabited type.",
        );
    }

    let (tag_scalar, tag_field, tag_encoding) = match &layout.variants {
        Variants::Single { index } => {
            let discr_val = layout
                .ty
                .discriminant_for_variant(fx.tcx, *index)
                .map_or(u128::from(index.as_u32()), |discr| discr.val);
            return CValue::const_val(fx, dest_layout, discr_val.into());
        }
        Variants::Multiple {
            tag,
            tag_field,
            tag_encoding,
            variants: _,
        } => (tag, *tag_field, tag_encoding),
    };

    let cast_to = fx.clif_type(dest_layout.ty).unwrap();

    // Read the tag/niche-encoded discriminant from memory.
    let tag = value.value_field(fx, mir::Field::new(tag_field));
    let tag = tag.load_scalar(fx);

    // Decode the discriminant (specifically if it's niche-encoded).
    match *tag_encoding {
        TagEncoding::Direct => {
            let signed = match tag_scalar.value {
                Int(_, signed) => signed,
                _ => false,
            };
            let val = clif_intcast(fx, tag, cast_to, signed);
            CValue::by_val(val, dest_layout)
        }
        TagEncoding::Niche {
            dataful_variant,
            ref niche_variants,
            niche_start,
        } => {
            // Rebase from niche values to discriminants, and check
            // whether the result is in range for the niche variants.

            // We first compute the "relative discriminant" (wrt `niche_variants`),
            // that is, if `n = niche_variants.end() - niche_variants.start()`,
            // we remap `niche_start..=niche_start + n` (which may wrap around)
            // to (non-wrap-around) `0..=n`, to be able to check whether the
            // discriminant corresponds to a niche variant with one comparison.
            // We also can't go directly to the (variant index) discriminant
            // and check that it is in the range `niche_variants`, because
            // that might not fit in the same type, on top of needing an extra
            // comparison (see also the comment on `let niche_discr`).
            let relative_discr = if niche_start == 0 {
                tag
            } else {
                // FIXME handle niche_start > i64::MAX
                fx.bcx
                    .ins()
                    .iadd_imm(tag, -i64::try_from(niche_start).unwrap())
            };
            let relative_max = niche_variants.end().as_u32() - niche_variants.start().as_u32();
            let is_niche = {
                codegen_icmp_imm(
                    fx,
                    IntCC::UnsignedLessThanOrEqual,
                    relative_discr,
                    i128::from(relative_max),
                )
            };

            // NOTE(eddyb) this addition needs to be performed on the final
            // type, in case the niche itself can't represent all variant
            // indices (e.g. `u8` niche with more than `256` variants,
            // but enough uninhabited variants so that the remaining variants
            // fit in the niche).
            // In other words, `niche_variants.end - niche_variants.start`
            // is representable in the niche, but `niche_variants.end`
            // might not be, in extreme cases.
            let niche_discr = {
                let relative_discr = if relative_max == 0 {
                    // HACK(eddyb) since we have only one niche, we know which
                    // one it is, and we can avoid having a dynamic value here.
                    fx.bcx.ins().iconst(cast_to, 0)
                } else {
                    clif_intcast(fx, relative_discr, cast_to, false)
                };
                fx.bcx
                    .ins()
                    .iadd_imm(relative_discr, i64::from(niche_variants.start().as_u32()))
            };

            let dataful_variant = fx
                .bcx
                .ins()
                .iconst(cast_to, i64::from(dataful_variant.as_u32()));
            let discr = fx.bcx.ins().select(is_niche, niche_discr, dataful_variant);
            CValue::by_val(discr, dest_layout)
        }
    }
}
