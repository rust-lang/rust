//! Compute the binary representation of structs, unions and enums

use std::{cmp, ops::Bound};

use base_db::ra_salsa::Cycle;
use hir_def::{
    data::adt::VariantData,
    layout::{Integer, ReprOptions, TargetDataLayout},
    AdtId, VariantId,
};
use intern::sym;
use rustc_index::IndexVec;
use smallvec::SmallVec;
use triomphe::Arc;

use crate::{
    db::HirDatabase,
    lang_items::is_unsafe_cell,
    layout::{field_ty, Layout, LayoutError, RustcEnumVariantIdx},
    Substitution, TraitEnvironment,
};

use super::LayoutCx;

pub(crate) fn struct_variant_idx() -> RustcEnumVariantIdx {
    RustcEnumVariantIdx(0)
}

pub fn layout_of_adt_query(
    db: &dyn HirDatabase,
    def: AdtId,
    subst: Substitution,
    trait_env: Arc<TraitEnvironment>,
) -> Result<Arc<Layout>, LayoutError> {
    let krate = trait_env.krate;
    let Ok(target) = db.target_data_layout(krate) else {
        return Err(LayoutError::TargetLayoutNotAvailable);
    };
    let dl = &*target;
    let cx = LayoutCx::new(dl);
    let handle_variant = |def: VariantId, var: &VariantData| {
        var.fields()
            .iter()
            .map(|(fd, _)| db.layout_of_ty(field_ty(db, def, fd, &subst), trait_env.clone()))
            .collect::<Result<Vec<_>, _>>()
    };
    let (variants, repr) = match def {
        AdtId::StructId(s) => {
            let data = db.struct_data(s);
            let mut r = SmallVec::<[_; 1]>::new();
            r.push(handle_variant(s.into(), &data.variant_data)?);
            (r, data.repr.unwrap_or_default())
        }
        AdtId::UnionId(id) => {
            let data = db.union_data(id);
            let mut r = SmallVec::new();
            r.push(handle_variant(id.into(), &data.variant_data)?);
            (r, data.repr.unwrap_or_default())
        }
        AdtId::EnumId(e) => {
            let data = db.enum_data(e);
            let r = data
                .variants
                .iter()
                .map(|&(v, _)| handle_variant(v.into(), &db.enum_variant_data(v).variant_data))
                .collect::<Result<SmallVec<_>, _>>()?;
            (r, data.repr.unwrap_or_default())
        }
    };
    let variants = variants
        .iter()
        .map(|it| it.iter().map(|it| &**it).collect::<Vec<_>>())
        .collect::<SmallVec<[_; 1]>>();
    let variants = variants.iter().map(|it| it.iter().collect()).collect::<IndexVec<_, _>>();
    let result = if matches!(def, AdtId::UnionId(..)) {
        cx.calc.layout_of_union(&repr, &variants)?
    } else {
        cx.calc.layout_of_struct_or_enum(
            &repr,
            &variants,
            matches!(def, AdtId::EnumId(..)),
            is_unsafe_cell(db, def),
            layout_scalar_valid_range(db, def),
            |min, max| repr_discr(dl, &repr, min, max).unwrap_or((Integer::I8, false)),
            variants.iter_enumerated().filter_map(|(id, _)| {
                let AdtId::EnumId(e) = def else { return None };
                let d = db.const_eval_discriminant(db.enum_data(e).variants[id.0].0).ok()?;
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
            !matches!(def, AdtId::EnumId(..))
                && variants
                    .iter()
                    .next()
                    .and_then(|it| it.iter().last().map(|it| !it.is_unsized()))
                    .unwrap_or(true),
        )?
    };
    Ok(Arc::new(result))
}

fn layout_scalar_valid_range(db: &dyn HirDatabase, def: AdtId) -> (Bound<u128>, Bound<u128>) {
    let attrs = db.attrs(def.into());
    let get = |name| {
        let attr = attrs.by_key(name).tt_values();
        for tree in attr {
            if let Some(it) = tree.token_trees.first() {
                let text = it.to_string().replace('_', "");
                let (text, base) = match text.as_bytes() {
                    [b'0', b'x', ..] => (&text[2..], 16),
                    [b'0', b'o', ..] => (&text[2..], 8),
                    [b'0', b'b', ..] => (&text[2..], 2),
                    _ => (&*text, 10),
                };

                if let Ok(it) = u128::from_str_radix(text, base) {
                    return Bound::Included(it);
                }
            }
        }
        Bound::Unbounded
    };
    (
        get(&sym::rustc_layout_scalar_valid_range_start),
        get(&sym::rustc_layout_scalar_valid_range_end),
    )
}

pub fn layout_of_adt_recover(
    _: &dyn HirDatabase,
    _: &Cycle,
    _: &AdtId,
    _: &Substitution,
    _: &Arc<TraitEnvironment>,
) -> Result<Arc<Layout>, LayoutError> {
    Err(LayoutError::RecursiveTypeWithoutIndirection)
}

/// Finds the appropriate Integer type and signedness for the given
/// signed discriminant range and `#[repr]` attribute.
/// N.B.: `u128` values above `i128::MAX` will be treated as signed, but
/// that shouldn't affect anything, other than maybe debuginfo.
fn repr_discr(
    dl: &TargetDataLayout,
    repr: &ReprOptions,
    min: i128,
    max: i128,
) -> Result<(Integer, bool), LayoutError> {
    // Theoretically, negative values could be larger in unsigned representation
    // than the unsigned representation of the signed minimum. However, if there
    // are any negative values, the only valid unsigned representation is u128
    // which can fit all i128 values, so the result remains unaffected.
    let unsigned_fit = Integer::fit_unsigned(cmp::max(min as u128, max as u128));
    let signed_fit = cmp::max(Integer::fit_signed(min), Integer::fit_signed(max));

    if let Some(ity) = repr.int {
        let discr = Integer::from_attr(dl, ity);
        let fit = if ity.is_signed() { signed_fit } else { unsigned_fit };
        if discr < fit {
            return Err(LayoutError::UserReprTooSmall);
        }
        return Ok((discr, ity.is_signed()));
    }

    let at_least = if repr.c() {
        // This is usually I32, however it can be different on some platforms,
        // notably hexagon and arm-none/thumb-none
        dl.c_enum_min_size
    } else {
        // repr(Rust) enums try to be as small as possible
        Integer::I8
    };

    // If there are no negative values, we can use the unsigned fit.
    Ok(if min >= 0 {
        (cmp::max(unsigned_fit, at_least), false)
    } else {
        (cmp::max(signed_fit, at_least), true)
    })
}
