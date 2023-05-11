//! Compute the binary representation of structs, unions and enums

use std::ops::Bound;

use hir_def::{
    adt::VariantData,
    layout::{Integer, IntegerExt, Layout, LayoutCalculator, LayoutError, RustcEnumVariantIdx},
    AdtId, EnumVariantId, HasModule, LocalEnumVariantId, VariantId,
};
use la_arena::RawIdx;
use smallvec::SmallVec;

use crate::{db::HirDatabase, lang_items::is_unsafe_cell, layout::field_ty, Substitution};

use super::{layout_of_ty, LayoutCx};

pub(crate) fn struct_variant_idx() -> RustcEnumVariantIdx {
    RustcEnumVariantIdx(LocalEnumVariantId::from_raw(RawIdx::from(0)))
}

pub fn layout_of_adt_query(
    db: &dyn HirDatabase,
    def: AdtId,
    subst: Substitution,
) -> Result<Layout, LayoutError> {
    let krate = def.module(db.upcast()).krate();
    let Some(target) = db.target_data_layout(krate) else { return Err(LayoutError::TargetLayoutNotAvailable) };
    let cx = LayoutCx { krate, target: &target };
    let dl = cx.current_data_layout();
    let handle_variant = |def: VariantId, var: &VariantData| {
        var.fields()
            .iter()
            .map(|(fd, _)| layout_of_ty(db, &field_ty(db, def, fd, &subst), cx.krate))
            .collect::<Result<Vec<_>, _>>()
    };
    let (variants, is_enum, is_union, repr) = match def {
        AdtId::StructId(s) => {
            let data = db.struct_data(s);
            let mut r = SmallVec::<[_; 1]>::new();
            r.push(handle_variant(s.into(), &data.variant_data)?);
            (r, false, false, data.repr.unwrap_or_default())
        }
        AdtId::UnionId(id) => {
            let data = db.union_data(id);
            let mut r = SmallVec::new();
            r.push(handle_variant(id.into(), &data.variant_data)?);
            (r, false, true, data.repr.unwrap_or_default())
        }
        AdtId::EnumId(e) => {
            let data = db.enum_data(e);
            let r = data
                .variants
                .iter()
                .map(|(idx, v)| {
                    handle_variant(
                        EnumVariantId { parent: e, local_id: idx }.into(),
                        &v.variant_data,
                    )
                })
                .collect::<Result<SmallVec<_>, _>>()?;
            (r, true, false, data.repr.unwrap_or_default())
        }
    };
    let variants =
        variants.iter().map(|x| x.iter().collect::<Vec<_>>()).collect::<SmallVec<[_; 1]>>();
    let variants = variants.iter().map(|x| x.iter().collect()).collect();
    if is_union {
        cx.layout_of_union(&repr, &variants).ok_or(LayoutError::Unknown)
    } else {
        cx.layout_of_struct_or_enum(
            &repr,
            &variants,
            is_enum,
            is_unsafe_cell(def, db),
            layout_scalar_valid_range(db, def),
            |min, max| Integer::repr_discr(&dl, &repr, min, max).unwrap_or((Integer::I8, false)),
            variants.iter_enumerated().filter_map(|(id, _)| {
                let AdtId::EnumId(e) = def else { return None };
                let d =
                    db.const_eval_discriminant(EnumVariantId { parent: e, local_id: id.0 }).ok()?;
                Some((id, d))
            }),
            // FIXME: The current code for niche-filling relies on variant indices
            // instead of actual discriminants, so enums with
            // explicit discriminants (RFC #2363) would misbehave and we should disable
            // niche optimization for them.
            // The code that do it in rustc:
            // repr.inhibit_enum_layout_opt() || def
            //     .variants()
            //     .iter_enumerated()
            //     .any(|(i, v)| v.discr != ty::VariantDiscr::Relative(i.as_u32()))
            repr.inhibit_enum_layout_opt(),
            !is_enum
                && variants
                    .iter()
                    .next()
                    .and_then(|x| x.last().map(|x| x.is_unsized()))
                    .unwrap_or(true),
        )
        .ok_or(LayoutError::SizeOverflow)
    }
}

fn layout_scalar_valid_range(db: &dyn HirDatabase, def: AdtId) -> (Bound<u128>, Bound<u128>) {
    let attrs = db.attrs(def.into());
    let get = |name| {
        let attr = attrs.by_key(name).tt_values();
        for tree in attr {
            if let Some(x) = tree.token_trees.first() {
                if let Ok(x) = x.to_string().parse() {
                    return Bound::Included(x);
                }
            }
        }
        Bound::Unbounded
    };
    (get("rustc_layout_scalar_valid_range_start"), get("rustc_layout_scalar_valid_range_end"))
}

pub fn layout_of_adt_recover(
    _: &dyn HirDatabase,
    _: &[String],
    _: &AdtId,
    _: &Substitution,
) -> Result<Layout, LayoutError> {
    user_error!("infinite sized recursive type");
}
