//! Compute the binary representation of structs, unions and enums

use std::{cmp, ops::Bound};

use hir_def::{
    AdtId, VariantId,
    attrs::AttrFlags,
    signatures::{StructFlags, VariantFields},
};
use rustc_abi::{Integer, ReprOptions, TargetDataLayout};
use rustc_index::IndexVec;
use smallvec::SmallVec;
use triomphe::Arc;

use crate::{
    ParamEnvAndCrate,
    db::HirDatabase,
    layout::{Layout, LayoutCx, LayoutError, field_ty},
    next_solver::GenericArgs,
};

pub fn layout_of_adt_query<'db>(
    db: &'db dyn HirDatabase,
    def: AdtId,
    args: GenericArgs<'db>,
    trait_env: ParamEnvAndCrate<'db>,
) -> Result<Arc<Layout>, LayoutError> {
    let krate = trait_env.krate;
    let Ok(target) = db.target_data_layout(krate) else {
        return Err(LayoutError::TargetLayoutNotAvailable);
    };
    let dl = &*target;
    let cx = LayoutCx::new(dl);
    let handle_variant = |def: VariantId, var: &VariantFields| {
        var.fields()
            .iter()
            .map(|(fd, _)| db.layout_of_ty(field_ty(db, def, fd, &args), trait_env))
            .collect::<Result<Vec<_>, _>>()
    };
    let (variants, repr, is_special_no_niche) = match def {
        AdtId::StructId(s) => {
            let sig = db.struct_signature(s);
            let mut r = SmallVec::<[_; 1]>::new();
            r.push(handle_variant(s.into(), s.fields(db))?);
            (
                r,
                AttrFlags::repr(db, s.into()).unwrap_or_default(),
                sig.flags.intersects(StructFlags::IS_UNSAFE_CELL | StructFlags::IS_UNSAFE_PINNED),
            )
        }
        AdtId::UnionId(id) => {
            let repr = AttrFlags::repr(db, id.into());
            let mut r = SmallVec::new();
            r.push(handle_variant(id.into(), id.fields(db))?);
            (r, repr.unwrap_or_default(), false)
        }
        AdtId::EnumId(e) => {
            let variants = e.enum_variants(db);
            let r = variants
                .variants
                .iter()
                .map(|&(v, _, _)| handle_variant(v.into(), v.fields(db)))
                .collect::<Result<SmallVec<_>, _>>()?;
            (r, AttrFlags::repr(db, e.into()).unwrap_or_default(), false)
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
            is_special_no_niche,
            layout_scalar_valid_range(db, def),
            |min, max| repr_discr(dl, &repr, min, max).unwrap_or((Integer::I8, false)),
            variants.iter_enumerated().filter_map(|(id, _)| {
                let AdtId::EnumId(e) = def else { return None };
                let d = db.const_eval_discriminant(e.enum_variants(db).variants[id.0].0).ok()?;
                Some((id, d))
            }),
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

pub(crate) fn layout_of_adt_cycle_result<'db>(
    _: &'db dyn HirDatabase,
    _def: AdtId,
    _args: GenericArgs<'db>,
    _trait_env: ParamEnvAndCrate<'db>,
) -> Result<Arc<Layout>, LayoutError> {
    Err(LayoutError::RecursiveTypeWithoutIndirection)
}

fn layout_scalar_valid_range(db: &dyn HirDatabase, def: AdtId) -> (Bound<u128>, Bound<u128>) {
    let range = AttrFlags::rustc_layout_scalar_valid_range(db, def);
    let get = |value| match value {
        Some(it) => Bound::Included(it),
        None => Bound::Unbounded,
    };
    (get(range.start), get(range.end))
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
