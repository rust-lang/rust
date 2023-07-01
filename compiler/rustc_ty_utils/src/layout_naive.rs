use rustc_middle::query::Providers;
use rustc_middle::ty::layout::{
    IntegerExt, LayoutCx, LayoutError, LayoutOf, NaiveAbi, NaiveLayout, TyAndNaiveLayout,
};
use rustc_middle::ty::{self, ReprOptions, Ty, TyCtxt, TypeVisitableExt};

use rustc_span::DUMMY_SP;
use rustc_target::abi::*;

use crate::layout::{compute_array_count, ptr_metadata_scalar};

pub fn provide(providers: &mut Providers) {
    *providers = Providers { naive_layout_of, ..*providers };
}

#[instrument(skip(tcx, query), level = "debug")]
fn naive_layout_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>,
) -> Result<TyAndNaiveLayout<'tcx>, &'tcx LayoutError<'tcx>> {
    let (param_env, ty) = query.into_parts();
    debug!(?ty);

    let param_env = param_env.with_reveal_all_normalized(tcx);
    let unnormalized_ty = ty;

    // FIXME: We might want to have two different versions of `layout_of`:
    // One that can be called after typecheck has completed and can use
    // `normalize_erasing_regions` here and another one that can be called
    // before typecheck has completed and uses `try_normalize_erasing_regions`.
    let ty = match tcx.try_normalize_erasing_regions(param_env, ty) {
        Ok(t) => t,
        Err(normalization_error) => {
            return Err(tcx
                .arena
                .alloc(LayoutError::NormalizationFailure(ty, normalization_error)));
        }
    };

    if ty != unnormalized_ty {
        // Ensure this layout is also cached for the normalized type.
        return tcx.naive_layout_of(param_env.and(ty));
    }

    let cx = LayoutCx { tcx, param_env };
    let layout = naive_layout_of_uncached(&cx, ty)?;
    Ok(TyAndNaiveLayout { ty, layout })
}

fn error<'tcx>(
    cx: &LayoutCx<'tcx, TyCtxt<'tcx>>,
    err: LayoutError<'tcx>,
) -> &'tcx LayoutError<'tcx> {
    cx.tcx.arena.alloc(err)
}

fn naive_layout_of_uncached<'tcx>(
    cx: &LayoutCx<'tcx, TyCtxt<'tcx>>,
    ty: Ty<'tcx>,
) -> Result<NaiveLayout, &'tcx LayoutError<'tcx>> {
    let tcx = cx.tcx;
    let dl = cx.data_layout();

    let scalar = |value: Primitive| NaiveLayout {
        abi: NaiveAbi::Scalar(value),
        size: value.size(dl),
        align: value.align(dl).abi,
        exact: true,
    };

    let univariant = |fields: &mut dyn Iterator<Item = Ty<'tcx>>,
                      repr: &ReprOptions|
     -> Result<NaiveLayout, &'tcx LayoutError<'tcx>> {
        if repr.pack.is_some() && repr.align.is_some() {
            cx.tcx.sess.delay_span_bug(DUMMY_SP, "struct cannot be packed and aligned");
            return Err(error(cx, LayoutError::Unknown(ty)));
        }

        let linear = repr.inhibit_struct_field_reordering_opt();
        let pack = repr.pack.unwrap_or(Align::MAX);
        let mut layout = NaiveLayout::EMPTY;

        for field in fields {
            let field = cx.naive_layout_of(field)?.packed(pack);
            if linear {
                layout = layout.pad_to_align(field.align);
            }
            layout = layout
                .concat(&field, dl)
                .ok_or_else(|| error(cx, LayoutError::SizeOverflow(ty)))?;
        }

        if let Some(align) = repr.align {
            layout = layout.align_to(align);
        }

        if linear {
            layout.abi = layout.abi.as_aggregate();
        }

        Ok(layout.pad_to_align(layout.align))
    };

    debug_assert!(!ty.has_non_region_infer());

    Ok(match *ty.kind() {
        // Basic scalars
        ty::Bool => scalar(Int(I8, false)),
        ty::Char => scalar(Int(I32, false)),
        ty::Int(ity) => scalar(Int(Integer::from_int_ty(dl, ity), true)),
        ty::Uint(ity) => scalar(Int(Integer::from_uint_ty(dl, ity), false)),
        ty::Float(fty) => scalar(match fty {
            ty::FloatTy::F32 => F32,
            ty::FloatTy::F64 => F64,
        }),
        ty::FnPtr(_) => scalar(Pointer(dl.instruction_address_space)),

        // The never type.
        ty::Never => NaiveLayout { abi: NaiveAbi::Uninhabited, ..NaiveLayout::EMPTY },

        // Potentially-wide pointers.
        ty::Ref(_, pointee, _) | ty::RawPtr(ty::TypeAndMut { ty: pointee, .. }) => {
            let data_ptr = scalar(Pointer(AddressSpace::DATA));
            if let Some(metadata) = ptr_metadata_scalar(cx, pointee)? {
                // Effectively a (ptr, meta) tuple.
                let l = data_ptr
                    .concat(&scalar(metadata.primitive()), dl)
                    .ok_or_else(|| error(cx, LayoutError::SizeOverflow(ty)))?;
                l.pad_to_align(l.align)
            } else {
                // No metadata, this is a thin pointer.
                data_ptr
            }
        }

        ty::Dynamic(_, _, ty::DynStar) => {
            let ptr = scalar(Pointer(AddressSpace::DATA));
            ptr.concat(&ptr, dl).ok_or_else(|| error(cx, LayoutError::SizeOverflow(ty)))?
        }

        // Arrays and slices.
        ty::Array(element, count) => {
            let count = compute_array_count(cx, count)
                .ok_or_else(|| error(cx, LayoutError::Unknown(ty)))?;
            let element = cx.naive_layout_of(element)?;
            NaiveLayout {
                abi: element.abi.as_aggregate(),
                size: element
                    .size
                    .checked_mul(count, cx)
                    .ok_or_else(|| error(cx, LayoutError::SizeOverflow(ty)))?,
                ..*element
            }
        }
        ty::Slice(element) => {
            let element = cx.naive_layout_of(element)?;
            NaiveLayout { abi: NaiveAbi::Unsized, size: Size::ZERO, ..*element }
        }

        ty::FnDef(..) => NaiveLayout::EMPTY,

        // Unsized types.
        ty::Str | ty::Dynamic(_, _, ty::Dyn) | ty::Foreign(..) => {
            NaiveLayout { abi: NaiveAbi::Unsized, ..NaiveLayout::EMPTY }
        }

        // FIXME(reference_niches): try to actually compute a reasonable layout estimate,
        // without duplicating too much code from `generator_layout`.
        ty::Generator(..) => NaiveLayout { exact: false, ..NaiveLayout::EMPTY },

        ty::Closure(_, ref substs) => {
            univariant(&mut substs.as_closure().upvar_tys(), &ReprOptions::default())?
        }

        ty::Tuple(tys) => univariant(&mut tys.iter(), &ReprOptions::default())?,

        ty::Adt(def, substs) if def.is_union() => {
            let repr = def.repr();
            let pack = repr.pack.unwrap_or(Align::MAX);
            if repr.pack.is_some() && repr.align.is_some() {
                cx.tcx.sess.delay_span_bug(DUMMY_SP, "union cannot be packed and aligned");
                return Err(error(cx, LayoutError::Unknown(ty)));
            }

            let mut layout = NaiveLayout::EMPTY;
            for f in &def.variants()[FIRST_VARIANT].fields {
                let field = cx.naive_layout_of(f.ty(tcx, substs))?;
                layout = layout.union(&field.packed(pack));
            }

            // Unions are always inhabited, and never scalar if `repr(C)`.
            if !matches!(layout.abi, NaiveAbi::Scalar(_)) || repr.inhibit_enum_layout_opt() {
                layout.abi = NaiveAbi::Sized;
            }

            if let Some(align) = repr.align {
                layout = layout.align_to(align);
            }
            layout.pad_to_align(layout.align)
        }

        ty::Adt(def, substs) => {
            let repr = def.repr();
            let base = NaiveLayout {
                // For simplicity, assume that any enum has its discriminant field (if it exists)
                // niched inside one of the variants; this will underestimate the size (and sometimes
                // alignment) of enums. We also doesn't compute exact alignment for SIMD structs.
                // FIXME(reference_niches): Be smarter here.
                // Also consider adding a special case for null-optimized enums, so that we can have
                // `Option<&T>: PointerLike` in generic contexts.
                exact: !def.is_enum() && !repr.simd(),
                // An ADT with no inhabited variants should have an uninhabited ABI.
                abi: NaiveAbi::Uninhabited,
                ..NaiveLayout::EMPTY
            };

            let layout = def.variants().iter().try_fold(base, |layout, v| {
                let mut fields = v.fields.iter().map(|f| f.ty(tcx, substs));
                let vlayout = univariant(&mut fields, &repr)?;
                Ok(layout.union(&vlayout))
            })?;
            layout.pad_to_align(layout.align)
        }

        // Types with no meaningful known layout.
        ty::Alias(..) => {
            // NOTE(eddyb) `layout_of` query should've normalized these away,
            // if that was possible, so there's no reason to try again here.
            return Err(error(cx, LayoutError::Unknown(ty)));
        }

        ty::Bound(..) | ty::GeneratorWitness(..) | ty::GeneratorWitnessMIR(..) | ty::Infer(_) => {
            bug!("Layout::compute: unexpected type `{}`", ty)
        }

        ty::Placeholder(..) | ty::Param(_) | ty::Error(_) => {
            return Err(error(cx, LayoutError::Unknown(ty)));
        }
    })
}
