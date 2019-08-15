use crate::prelude::*;

pub fn codegen_set_discriminant<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    place: CPlace<'tcx>,
    variant_index: VariantIdx,
) {
    let layout = place.layout();
    if layout.for_variant(&*fx, variant_index).abi == layout::Abi::Uninhabited {
        return;
    }
    match layout.variants {
        layout::Variants::Single { index } => {
            assert_eq!(index, variant_index);
        }
        layout::Variants::Multiple {
            discr: _,
            discr_index,
            discr_kind: layout::DiscriminantKind::Tag,
            variants: _,
        } => {
            let ptr = place.place_field(fx, mir::Field::new(discr_index));
            let to = layout
                .ty
                .discriminant_for_variant(fx.tcx, variant_index)
                .unwrap()
                .val;
            let discr = CValue::const_val(fx, ptr.layout().ty, to);
            ptr.write_cvalue(fx, discr);
        }
        layout::Variants::Multiple {
            discr: _,
            discr_index,
            discr_kind: layout::DiscriminantKind::Niche {
                dataful_variant,
                ref niche_variants,
                niche_start,
            },
            variants: _,
        } => {
            if variant_index != dataful_variant {
                let niche = place.place_field(fx, mir::Field::new(discr_index));
                //let niche_llty = niche.layout.immediate_llvm_type(bx.cx);
                let niche_value =
                    ((variant_index.as_u32() - niche_variants.start().as_u32()) as u128)
                        .wrapping_add(niche_start);
                // FIXME(eddyb) Check the actual primitive type here.
                let niche_llval = if niche_value == 0 {
                    CValue::const_val(fx, niche.layout().ty, 0)
                } else {
                    CValue::const_val(fx, niche.layout().ty, niche_value)
                };
                niche.write_cvalue(fx, niche_llval);
            }
        }
    }
}

pub fn codegen_get_discriminant<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    place: CPlace<'tcx>,
    dest_layout: TyLayout<'tcx>,
) -> CValue<'tcx> {
    let layout = place.layout();

    if layout.abi == layout::Abi::Uninhabited {
        return trap_unreachable_ret_value(fx, dest_layout, "[panic] Tried to get discriminant for uninhabited type.");
    }

    let (discr_scalar, discr_index, discr_kind) = match &layout.variants {
        layout::Variants::Single { index } => {
            let discr_val = layout
                .ty
                .ty_adt_def()
                .map_or(u128::from(index.as_u32()), |def| {
                    def.discriminant_for_variant(fx.tcx, *index).val
                });
            return CValue::const_val(fx, dest_layout.ty, discr_val);
        }
        layout::Variants::Multiple { discr, discr_index, discr_kind, variants: _ } => {
            (discr, *discr_index, discr_kind)
        }
    };

    let discr = place.place_field(fx, mir::Field::new(discr_index)).to_cvalue(fx);
    let discr_ty = discr.layout().ty;
    let lldiscr = discr.load_scalar(fx);
    match discr_kind {
        layout::DiscriminantKind::Tag => {
            let signed = match discr_scalar.value {
                layout::Int(_, signed) => signed,
                _ => false,
            };
            let val = clif_intcast(fx, lldiscr, fx.clif_type(dest_layout.ty).unwrap(), signed);
            return CValue::by_val(val, dest_layout);
        }
        layout::DiscriminantKind::Niche {
            dataful_variant,
            ref niche_variants,
            niche_start,
        } => {
            let niche_llty = fx.clif_type(discr_ty).unwrap();
            let dest_clif_ty = fx.clif_type(dest_layout.ty).unwrap();
            if niche_variants.start() == niche_variants.end() {
                let b = codegen_icmp_imm(fx, IntCC::Equal, lldiscr, *niche_start as i128);
                let if_true = fx
                    .bcx
                    .ins()
                    .iconst(dest_clif_ty, niche_variants.start().as_u32() as i64);
                let if_false = fx
                    .bcx
                    .ins()
                    .iconst(dest_clif_ty, dataful_variant.as_u32() as i64);
                let val = fx.bcx.ins().select(b, if_true, if_false);
                return CValue::by_val(val, dest_layout);
            } else {
                // Rebase from niche values to discriminant values.
                let delta = niche_start.wrapping_sub(niche_variants.start().as_u32() as u128);
                let delta = fx.bcx.ins().iconst(niche_llty, delta as u64 as i64);
                let lldiscr = fx.bcx.ins().isub(lldiscr, delta);
                let b = codegen_icmp_imm(
                    fx,
                    IntCC::UnsignedLessThanOrEqual,
                    lldiscr,
                    i128::from(niche_variants.end().as_u32()),
                );
                let if_true =
                    clif_intcast(fx, lldiscr, fx.clif_type(dest_layout.ty).unwrap(), false);
                let if_false = fx
                    .bcx
                    .ins()
                    .iconst(dest_clif_ty, dataful_variant.as_u32() as i64);
                let val = fx.bcx.ins().select(b, if_true, if_false);
                return CValue::by_val(val, dest_layout);
            }
        }
    }
}
