use rustc_middle::query::Providers;
use rustc_middle::ty::layout::{
    IntegerExt, LayoutCx, LayoutError, LayoutOf, NaiveAbi, NaiveLayout, NaiveNiches,
    TyAndNaiveLayout,
};
use rustc_middle::ty::{self, ReprOptions, Ty, TyCtxt, TypeVisitableExt};
use rustc_span::DUMMY_SP;
use rustc_target::abi::*;

use std::ops::Bound;

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

    let scalar = |niched: bool, value: Primitive| NaiveLayout {
        abi: NaiveAbi::Scalar(value),
        niches: if niched { NaiveNiches::Some } else { NaiveNiches::None },
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
        ty::Bool => scalar(true, Int(I8, false)),
        ty::Char => scalar(true, Int(I32, false)),
        ty::Int(ity) => scalar(false, Int(Integer::from_int_ty(dl, ity), true)),
        ty::Uint(ity) => scalar(false, Int(Integer::from_uint_ty(dl, ity), false)),
        ty::Float(fty) => scalar(
            false,
            match fty {
                ty::FloatTy::F32 => F32,
                ty::FloatTy::F64 => F64,
            },
        ),
        ty::FnPtr(_) => scalar(true, Pointer(dl.instruction_address_space)),

        // The never type.
        ty::Never => NaiveLayout { abi: NaiveAbi::Uninhabited, ..NaiveLayout::EMPTY },

        // Potentially-wide pointers.
        ty::Ref(_, pointee, _) | ty::RawPtr(ty::TypeAndMut { ty: pointee, .. }) => {
            let data_ptr = scalar(!ty.is_unsafe_ptr(), Pointer(AddressSpace::DATA));
            if let Some(metadata) = ptr_metadata_scalar(cx, pointee)? {
                // Effectively a (ptr, meta) tuple.
                let meta = scalar(!metadata.is_always_valid(dl), metadata.primitive());
                let l = data_ptr
                    .concat(&meta, dl)
                    .ok_or_else(|| error(cx, LayoutError::SizeOverflow(ty)))?;
                l.pad_to_align(l.align)
            } else {
                // No metadata, this is a thin pointer.
                data_ptr
            }
        }

        ty::Dynamic(_, _, ty::DynStar) => {
            let ptr = scalar(false, Pointer(AddressSpace::DATA));
            let vtable = scalar(true, Pointer(AddressSpace::DATA));
            ptr.concat(&vtable, dl).ok_or_else(|| error(cx, LayoutError::SizeOverflow(ty)))?
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
                niches: if count == 0 { NaiveNiches::None } else { element.niches },
                ..*element
            }
        }
        ty::Slice(element) => NaiveLayout {
            abi: NaiveAbi::Unsized,
            size: Size::ZERO,
            niches: NaiveNiches::None,
            ..*cx.naive_layout_of(element)?
        },

        ty::FnDef(..) => NaiveLayout::EMPTY,

        // Unsized types.
        ty::Str | ty::Dynamic(_, _, ty::Dyn) | ty::Foreign(..) => {
            NaiveLayout { abi: NaiveAbi::Unsized, ..NaiveLayout::EMPTY }
        }

        // FIXME(reference_niches): try to actually compute a reasonable layout estimate,
        // without duplicating too much code from `generator_layout`.
        ty::Generator(..) => {
            NaiveLayout { exact: false, niches: NaiveNiches::Maybe, ..NaiveLayout::EMPTY }
        }

        ty::Closure(_, ref substs) => {
            univariant(&mut substs.as_closure().upvar_tys(), &ReprOptions::default())?
        }

        ty::Tuple(tys) => univariant(&mut tys.iter(), &ReprOptions::default())?,

        ty::Adt(def, substs) if def.is_union() => {
            assert_eq!(def.variants().len(), 1, "union should have a single variant");
            let repr = def.repr();
            let pack = repr.pack.unwrap_or(Align::MAX);
            if repr.pack.is_some() && repr.align.is_some() {
                cx.tcx.sess.delay_span_bug(DUMMY_SP, "union cannot be packed and aligned");
                return Err(error(cx, LayoutError::Unknown(ty)));
            }

            let mut layout = NaiveLayout {
                // Unions never have niches.
                niches: NaiveNiches::None,
                ..NaiveLayout::EMPTY
            };

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
            let mut layout = NaiveLayout {
                // An ADT with no inhabited variants should have an uninhabited ABI.
                abi: NaiveAbi::Uninhabited,
                ..NaiveLayout::EMPTY
            };

            let mut empty_variants = 0;
            for v in def.variants() {
                let mut fields = v.fields.iter().map(|f| f.ty(tcx, substs));
                let vlayout = univariant(&mut fields, &repr)?;

                if vlayout.size == Size::ZERO && vlayout.exact {
                    empty_variants += 1;
                } else {
                    // Remember the niches of the last seen variant.
                    layout.niches = vlayout.niches;
                }

                layout = layout.union(&vlayout);
            }

            if def.is_enum() {
                let may_need_discr = match def.variants().len() {
                    0 | 1 => false,
                    // Simple Option-like niche optimization.
                    // Handling this special case allows enums like `Option<&T>`
                    // to be recognized as `PointerLike` and to be transmutable
                    // in generic contexts.
                    2 if empty_variants == 1 && layout.niches == NaiveNiches::Some => {
                        layout.niches = NaiveNiches::Maybe; // fill up the niche.
                        false
                    }
                    _ => true,
                };

                if may_need_discr || repr.inhibit_enum_layout_opt() {
                    // For simplicity, assume that the discriminant always get niched.
                    // This will be wrong in many cases, which will cause the size (and
                    // sometimes the alignment) to be underestimated.
                    // FIXME(reference_niches): Be smarter here.
                    layout.niches = NaiveNiches::Maybe;
                    layout = layout.inexact();
                }
            } else {
                assert_eq!(def.variants().len(), 1, "struct should have a single variant");

                // We don't compute exact alignment for SIMD structs.
                if repr.simd() {
                    layout = layout.inexact();
                }

                // `UnsafeCell` hides all niches.
                if def.is_unsafe_cell() {
                    layout.niches = NaiveNiches::None;
                }
            }

            let valid_range = tcx.layout_scalar_valid_range(def.did());
            if valid_range != (Bound::Unbounded, Bound::Unbounded) {
                let get = |bound, default| match bound {
                    Bound::Unbounded => default,
                    Bound::Included(v) => v,
                    Bound::Excluded(_) => bug!("exclusive `layout_scalar_valid_range` bound"),
                };

                let valid_range = WrappingRange {
                    start: get(valid_range.0, 0),
                    // FIXME: this is wrong for scalar-pair ABIs. Fortunately, the
                    // only type this could currently affect is`NonNull<T: !Sized>`,
                    // and the `NaiveNiches` result still ends up correct.
                    end: get(valid_range.1, layout.size.unsigned_int_max()),
                };
                assert!(
                    valid_range.is_in_range_for(layout.size),
                    "`layout_scalar_valid_range` values are out of bounds",
                );
                if !valid_range.is_full_for(layout.size) {
                    layout.niches = NaiveNiches::Some;
                }
            }

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
