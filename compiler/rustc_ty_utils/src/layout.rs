use hir::def_id::DefId;
use rustc_hir as hir;
use rustc_index::bit_set::BitSet;
use rustc_index::{IndexSlice, IndexVec};
use rustc_middle::mir::{GeneratorLayout, GeneratorSavedLocal};
use rustc_middle::query::Providers;
use rustc_middle::ty::layout::{
    IntegerExt, LayoutCx, LayoutError, LayoutOf, TyAndLayout, MAX_SIMD_LANES,
};
use rustc_middle::ty::{
    self, AdtDef, EarlyBinder, GenericArgsRef, ReprOptions, Ty, TyCtxt, TypeVisitableExt,
};
use rustc_session::{DataTypeKind, FieldInfo, FieldKind, SizeKind, VariantInfo};
use rustc_span::symbol::Symbol;
use rustc_span::DUMMY_SP;
use rustc_target::abi::*;

use std::fmt::Debug;
use std::iter;

use crate::errors::{
    MultipleArrayFieldsSimdType, NonPrimitiveSimdType, OversizedSimdType, ZeroLengthSimdType,
};
use crate::layout_sanity_check::sanity_check_layout;

pub fn provide(providers: &mut Providers) {
    *providers = Providers { layout_of, ..*providers };
}

#[instrument(skip(tcx, query), level = "debug")]
fn layout_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>,
) -> Result<TyAndLayout<'tcx>, &'tcx LayoutError<'tcx>> {
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
        return tcx.layout_of(param_env.and(ty));
    }

    let cx = LayoutCx { tcx, param_env };

    let layout = layout_of_uncached(&cx, ty)?;
    let layout = TyAndLayout { ty, layout };

    record_layout_for_printing(&cx, layout);

    sanity_check_layout(&cx, &layout);

    Ok(layout)
}

fn error<'tcx>(
    cx: &LayoutCx<'tcx, TyCtxt<'tcx>>,
    err: LayoutError<'tcx>,
) -> &'tcx LayoutError<'tcx> {
    cx.tcx.arena.alloc(err)
}

fn univariant_uninterned<'tcx>(
    cx: &LayoutCx<'tcx, TyCtxt<'tcx>>,
    ty: Ty<'tcx>,
    fields: &IndexSlice<FieldIdx, Layout<'_>>,
    repr: &ReprOptions,
    kind: StructKind,
) -> Result<LayoutS, &'tcx LayoutError<'tcx>> {
    let dl = cx.data_layout();
    let pack = repr.pack;
    if pack.is_some() && repr.align.is_some() {
        cx.tcx.sess.delay_span_bug(DUMMY_SP, "struct cannot be packed and aligned");
        return Err(cx.tcx.arena.alloc(LayoutError::Unknown(ty)));
    }

    cx.univariant(dl, fields, repr, kind).ok_or_else(|| error(cx, LayoutError::SizeOverflow(ty)))
}

fn layout_of_uncached<'tcx>(
    cx: &LayoutCx<'tcx, TyCtxt<'tcx>>,
    ty: Ty<'tcx>,
) -> Result<Layout<'tcx>, &'tcx LayoutError<'tcx>> {
    let tcx = cx.tcx;
    let param_env = cx.param_env;
    let dl = cx.data_layout();
    let scalar_unit = |value: Primitive| {
        let size = value.size(dl);
        assert!(size.bits() <= 128);
        Scalar::Initialized { value, valid_range: WrappingRange::full(size) }
    };
    let scalar = |value: Primitive| tcx.mk_layout(LayoutS::scalar(cx, scalar_unit(value)));

    let univariant = |fields: &IndexSlice<FieldIdx, Layout<'_>>, repr: &ReprOptions, kind| {
        Ok(tcx.mk_layout(univariant_uninterned(cx, ty, fields, repr, kind)?))
    };
    debug_assert!(!ty.has_non_region_infer());

    Ok(match *ty.kind() {
        // Basic scalars.
        ty::Bool => tcx.mk_layout(LayoutS::scalar(
            cx,
            Scalar::Initialized {
                value: Int(I8, false),
                valid_range: WrappingRange { start: 0, end: 1 },
            },
        )),
        ty::Char => tcx.mk_layout(LayoutS::scalar(
            cx,
            Scalar::Initialized {
                value: Int(I32, false),
                valid_range: WrappingRange { start: 0, end: 0x10FFFF },
            },
        )),
        ty::Int(ity) => scalar(Int(Integer::from_int_ty(dl, ity), true)),
        ty::Uint(ity) => scalar(Int(Integer::from_uint_ty(dl, ity), false)),
        ty::Float(fty) => scalar(match fty {
            ty::FloatTy::F32 => F32,
            ty::FloatTy::F64 => F64,
        }),
        ty::FnPtr(_) => {
            let mut ptr = scalar_unit(Pointer(dl.instruction_address_space));
            ptr.valid_range_mut().start = 1;
            tcx.mk_layout(LayoutS::scalar(cx, ptr))
        }

        // The never type.
        ty::Never => tcx.mk_layout(cx.layout_of_never_type()),

        // Potentially-wide pointers.
        ty::Ref(_, pointee, _) | ty::RawPtr(ty::TypeAndMut { ty: pointee, .. }) => {
            let mut data_ptr = scalar_unit(Pointer(AddressSpace::DATA));
            if !ty.is_unsafe_ptr() {
                data_ptr.valid_range_mut().start = 1;
            }

            let pointee = tcx.normalize_erasing_regions(param_env, pointee);
            if pointee.is_sized(tcx, param_env) {
                return Ok(tcx.mk_layout(LayoutS::scalar(cx, data_ptr)));
            }

            let metadata = if let Some(metadata_def_id) = tcx.lang_items().metadata_type()
                // Projection eagerly bails out when the pointee references errors,
                // fall back to structurally deducing metadata.
                && !pointee.references_error()
            {
                let pointee_metadata = Ty::new_projection(tcx,metadata_def_id, [pointee]);
                let metadata_ty = match tcx.try_normalize_erasing_regions(
                    param_env,
                    pointee_metadata,
                ) {
                    Ok(metadata_ty) => metadata_ty,
                    Err(mut err) => {
                        // Usually `<Ty as Pointee>::Metadata` can't be normalized because
                        // its struct tail cannot be normalized either, so try to get a
                        // more descriptive layout error here, which will lead to less confusing
                        // diagnostics.
                        match tcx.try_normalize_erasing_regions(
                            param_env,
                            tcx.struct_tail_without_normalization(pointee),
                        ) {
                            Ok(_) => {},
                            Err(better_err) => {
                                err = better_err;
                            }
                        }
                        return Err(error(cx, LayoutError::NormalizationFailure(pointee, err)));
                    },
                };

                let metadata_layout = cx.layout_of(metadata_ty)?;
                // If the metadata is a 1-zst, then the pointer is thin.
                if metadata_layout.is_zst() && metadata_layout.align.abi.bytes() == 1 {
                    return Ok(tcx.mk_layout(LayoutS::scalar(cx, data_ptr)));
                }

                let Abi::Scalar(metadata) = metadata_layout.abi else {
                    return Err(error(cx, LayoutError::Unknown(pointee)));
                };

                metadata
            } else {
                let unsized_part = tcx.struct_tail_erasing_lifetimes(pointee, param_env);

                match unsized_part.kind() {
                    ty::Foreign(..) => {
                        return Ok(tcx.mk_layout(LayoutS::scalar(cx, data_ptr)));
                    }
                    ty::Slice(_) | ty::Str => scalar_unit(Int(dl.ptr_sized_integer(), false)),
                    ty::Dynamic(..) => {
                        let mut vtable = scalar_unit(Pointer(AddressSpace::DATA));
                        vtable.valid_range_mut().start = 1;
                        vtable
                    }
                    _ => {
                        return Err(error(cx, LayoutError::Unknown(pointee)));
                    }
                }
            };

            // Effectively a (ptr, meta) tuple.
            tcx.mk_layout(cx.scalar_pair(data_ptr, metadata))
        }

        ty::Dynamic(_, _, ty::DynStar) => {
            let mut data = scalar_unit(Pointer(AddressSpace::DATA));
            data.valid_range_mut().start = 0;
            let mut vtable = scalar_unit(Pointer(AddressSpace::DATA));
            vtable.valid_range_mut().start = 1;
            tcx.mk_layout(cx.scalar_pair(data, vtable))
        }

        // Arrays and slices.
        ty::Array(element, mut count) => {
            if count.has_projections() {
                count = tcx.normalize_erasing_regions(param_env, count);
                if count.has_projections() {
                    return Err(error(cx, LayoutError::Unknown(ty)));
                }
            }

            let count = count
                .try_eval_target_usize(tcx, param_env)
                .ok_or_else(|| error(cx, LayoutError::Unknown(ty)))?;
            let element = cx.layout_of(element)?;
            let size = element
                .size
                .checked_mul(count, dl)
                .ok_or_else(|| error(cx, LayoutError::SizeOverflow(ty)))?;

            let abi = if count != 0 && ty.is_privately_uninhabited(tcx, param_env) {
                Abi::Uninhabited
            } else {
                Abi::Aggregate { sized: true }
            };

            let largest_niche = if count != 0 { element.largest_niche } else { None };

            tcx.mk_layout(LayoutS {
                variants: Variants::Single { index: FIRST_VARIANT },
                fields: FieldsShape::Array { stride: element.size, count },
                abi,
                largest_niche,
                align: element.align,
                size,
                max_repr_align: None,
                unadjusted_abi_align: element.align.abi,
            })
        }
        ty::Slice(element) => {
            let element = cx.layout_of(element)?;
            tcx.mk_layout(LayoutS {
                variants: Variants::Single { index: FIRST_VARIANT },
                fields: FieldsShape::Array { stride: element.size, count: 0 },
                abi: Abi::Aggregate { sized: false },
                largest_niche: None,
                align: element.align,
                size: Size::ZERO,
                max_repr_align: None,
                unadjusted_abi_align: element.align.abi,
            })
        }
        ty::Str => tcx.mk_layout(LayoutS {
            variants: Variants::Single { index: FIRST_VARIANT },
            fields: FieldsShape::Array { stride: Size::from_bytes(1), count: 0 },
            abi: Abi::Aggregate { sized: false },
            largest_niche: None,
            align: dl.i8_align,
            size: Size::ZERO,
            max_repr_align: None,
            unadjusted_abi_align: dl.i8_align.abi,
        }),

        // Odd unit types.
        ty::FnDef(..) => {
            univariant(IndexSlice::empty(), &ReprOptions::default(), StructKind::AlwaysSized)?
        }
        ty::Dynamic(_, _, ty::Dyn) | ty::Foreign(..) => {
            let mut unit = univariant_uninterned(
                cx,
                ty,
                IndexSlice::empty(),
                &ReprOptions::default(),
                StructKind::AlwaysSized,
            )?;
            match unit.abi {
                Abi::Aggregate { ref mut sized } => *sized = false,
                _ => bug!(),
            }
            tcx.mk_layout(unit)
        }

        ty::Generator(def_id, args, _) => generator_layout(cx, ty, def_id, args)?,

        ty::Closure(_, ref args) => {
            let tys = args.as_closure().upvar_tys();
            univariant(
                &tys.map(|ty| Ok(cx.layout_of(ty)?.layout)).try_collect::<IndexVec<_, _>>()?,
                &ReprOptions::default(),
                StructKind::AlwaysSized,
            )?
        }

        ty::Tuple(tys) => {
            let kind =
                if tys.len() == 0 { StructKind::AlwaysSized } else { StructKind::MaybeUnsized };

            univariant(
                &tys.iter().map(|k| Ok(cx.layout_of(k)?.layout)).try_collect::<IndexVec<_, _>>()?,
                &ReprOptions::default(),
                kind,
            )?
        }

        // SIMD vector types.
        ty::Adt(def, args) if def.repr().simd() => {
            if !def.is_struct() {
                // Should have yielded E0517 by now.
                tcx.sess.delay_span_bug(
                    DUMMY_SP,
                    "#[repr(simd)] was applied to an ADT that is not a struct",
                );
                return Err(error(cx, LayoutError::Unknown(ty)));
            }

            let fields = &def.non_enum_variant().fields;

            // Supported SIMD vectors are homogeneous ADTs with at least one field:
            //
            // * #[repr(simd)] struct S(T, T, T, T);
            // * #[repr(simd)] struct S { x: T, y: T, z: T, w: T }
            // * #[repr(simd)] struct S([T; 4])
            //
            // where T is a primitive scalar (integer/float/pointer).

            // SIMD vectors with zero fields are not supported.
            // (should be caught by typeck)
            if fields.is_empty() {
                tcx.sess.emit_fatal(ZeroLengthSimdType { ty })
            }

            // Type of the first ADT field:
            let f0_ty = fields[FieldIdx::from_u32(0)].ty(tcx, args);

            // Heterogeneous SIMD vectors are not supported:
            // (should be caught by typeck)
            for fi in fields {
                if fi.ty(tcx, args) != f0_ty {
                    tcx.sess.delay_span_bug(
                        DUMMY_SP,
                        "#[repr(simd)] was applied to an ADT with heterogeneous field type",
                    );
                    return Err(error(cx, LayoutError::Unknown(ty)));
                }
            }

            // The element type and number of elements of the SIMD vector
            // are obtained from:
            //
            // * the element type and length of the single array field, if
            // the first field is of array type, or
            //
            // * the homogeneous field type and the number of fields.
            let (e_ty, e_len, is_array) = if let ty::Array(e_ty, _) = f0_ty.kind() {
                // First ADT field is an array:

                // SIMD vectors with multiple array fields are not supported:
                // Can't be caught by typeck with a generic simd type.
                if def.non_enum_variant().fields.len() != 1 {
                    tcx.sess.emit_fatal(MultipleArrayFieldsSimdType { ty });
                }

                // Extract the number of elements from the layout of the array field:
                let FieldsShape::Array { count, .. } = cx.layout_of(f0_ty)?.layout.fields() else {
                    return Err(error(cx, LayoutError::Unknown(ty)));
                };

                (*e_ty, *count, true)
            } else {
                // First ADT field is not an array:
                (f0_ty, def.non_enum_variant().fields.len() as _, false)
            };

            // SIMD vectors of zero length are not supported.
            // Additionally, lengths are capped at 2^16 as a fixed maximum backends must
            // support.
            //
            // Can't be caught in typeck if the array length is generic.
            if e_len == 0 {
                tcx.sess.emit_fatal(ZeroLengthSimdType { ty });
            } else if e_len > MAX_SIMD_LANES {
                tcx.sess.emit_fatal(OversizedSimdType { ty, max_lanes: MAX_SIMD_LANES });
            }

            // Compute the ABI of the element type:
            let e_ly = cx.layout_of(e_ty)?;
            let Abi::Scalar(e_abi) = e_ly.abi else {
                // This error isn't caught in typeck, e.g., if
                // the element type of the vector is generic.
                tcx.sess.emit_fatal(NonPrimitiveSimdType { ty, e_ty });
            };

            // Compute the size and alignment of the vector:
            let size = e_ly
                .size
                .checked_mul(e_len, dl)
                .ok_or_else(|| error(cx, LayoutError::SizeOverflow(ty)))?;
            let align = dl.vector_align(size);
            let size = size.align_to(align.abi);

            // Compute the placement of the vector fields:
            let fields = if is_array {
                FieldsShape::Arbitrary { offsets: [Size::ZERO].into(), memory_index: [0].into() }
            } else {
                FieldsShape::Array { stride: e_ly.size, count: e_len }
            };

            tcx.mk_layout(LayoutS {
                variants: Variants::Single { index: FIRST_VARIANT },
                fields,
                abi: Abi::Vector { element: e_abi, count: e_len },
                largest_niche: e_ly.largest_niche,
                size,
                align,
                max_repr_align: None,
                unadjusted_abi_align: align.abi,
            })
        }

        // ADTs.
        ty::Adt(def, args) => {
            // Cache the field layouts.
            let variants = def
                .variants()
                .iter()
                .map(|v| {
                    v.fields
                        .iter()
                        .map(|field| Ok(cx.layout_of(field.ty(tcx, args))?.layout))
                        .try_collect::<IndexVec<_, _>>()
                })
                .try_collect::<IndexVec<VariantIdx, _>>()?;

            if def.is_union() {
                if def.repr().pack.is_some() && def.repr().align.is_some() {
                    cx.tcx.sess.delay_span_bug(
                        tcx.def_span(def.did()),
                        "union cannot be packed and aligned",
                    );
                    return Err(error(cx, LayoutError::Unknown(ty)));
                }

                return Ok(tcx.mk_layout(
                    cx.layout_of_union(&def.repr(), &variants)
                        .ok_or_else(|| error(cx, LayoutError::Unknown(ty)))?,
                ));
            }

            let get_discriminant_type =
                |min, max| Integer::repr_discr(tcx, ty, &def.repr(), min, max);

            let discriminants_iter = || {
                def.is_enum()
                    .then(|| def.discriminants(tcx).map(|(v, d)| (v, d.val as i128)))
                    .into_iter()
                    .flatten()
            };

            let dont_niche_optimize_enum = def.repr().inhibit_enum_layout_opt()
                || def
                    .variants()
                    .iter_enumerated()
                    .any(|(i, v)| v.discr != ty::VariantDiscr::Relative(i.as_u32()));

            let maybe_unsized = def.is_struct()
                && def.non_enum_variant().tail_opt().is_some_and(|last_field| {
                    let param_env = tcx.param_env(def.did());
                    !tcx.type_of(last_field.did).instantiate_identity().is_sized(tcx, param_env)
                });

            let Some(layout) = cx.layout_of_struct_or_enum(
                &def.repr(),
                &variants,
                def.is_enum(),
                def.is_unsafe_cell(),
                tcx.layout_scalar_valid_range(def.did()),
                get_discriminant_type,
                discriminants_iter(),
                dont_niche_optimize_enum,
                !maybe_unsized,
            ) else {
                return Err(error(cx, LayoutError::SizeOverflow(ty)));
            };

            // If the struct tail is sized and can be unsized, check that unsizing doesn't move the fields around.
            if cfg!(debug_assertions)
                && maybe_unsized
                && def.non_enum_variant().tail().ty(tcx, args).is_sized(tcx, cx.param_env)
            {
                let mut variants = variants;
                let tail_replacement = cx.layout_of(Ty::new_slice(tcx, tcx.types.u8)).unwrap();
                *variants[FIRST_VARIANT].raw.last_mut().unwrap() = tail_replacement.layout;

                let Some(unsized_layout) = cx.layout_of_struct_or_enum(
                    &def.repr(),
                    &variants,
                    def.is_enum(),
                    def.is_unsafe_cell(),
                    tcx.layout_scalar_valid_range(def.did()),
                    get_discriminant_type,
                    discriminants_iter(),
                    dont_niche_optimize_enum,
                    !maybe_unsized,
                ) else {
                    bug!("failed to compute unsized layout of {ty:?}");
                };

                let FieldsShape::Arbitrary { offsets: sized_offsets, .. } = &layout.fields else {
                    bug!("unexpected FieldsShape for sized layout of {ty:?}: {:?}", layout.fields);
                };
                let FieldsShape::Arbitrary { offsets: unsized_offsets, .. } =
                    &unsized_layout.fields
                else {
                    bug!(
                        "unexpected FieldsShape for unsized layout of {ty:?}: {:?}",
                        unsized_layout.fields
                    );
                };

                let (sized_tail, sized_fields) = sized_offsets.raw.split_last().unwrap();
                let (unsized_tail, unsized_fields) = unsized_offsets.raw.split_last().unwrap();

                if sized_fields != unsized_fields {
                    bug!("unsizing {ty:?} changed field order!\n{layout:?}\n{unsized_layout:?}");
                }

                if sized_tail < unsized_tail {
                    bug!("unsizing {ty:?} moved tail backwards!\n{layout:?}\n{unsized_layout:?}");
                }
            }

            tcx.mk_layout(layout)
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

/// Overlap eligibility and variant assignment for each GeneratorSavedLocal.
#[derive(Clone, Debug, PartialEq)]
enum SavedLocalEligibility {
    Unassigned,
    Assigned(VariantIdx),
    Ineligible(Option<FieldIdx>),
}

// When laying out generators, we divide our saved local fields into two
// categories: overlap-eligible and overlap-ineligible.
//
// Those fields which are ineligible for overlap go in a "prefix" at the
// beginning of the layout, and always have space reserved for them.
//
// Overlap-eligible fields are only assigned to one variant, so we lay
// those fields out for each variant and put them right after the
// prefix.
//
// Finally, in the layout details, we point to the fields from the
// variants they are assigned to. It is possible for some fields to be
// included in multiple variants. No field ever "moves around" in the
// layout; its offset is always the same.
//
// Also included in the layout are the upvars and the discriminant.
// These are included as fields on the "outer" layout; they are not part
// of any variant.

/// Compute the eligibility and assignment of each local.
fn generator_saved_local_eligibility(
    info: &GeneratorLayout<'_>,
) -> (BitSet<GeneratorSavedLocal>, IndexVec<GeneratorSavedLocal, SavedLocalEligibility>) {
    use SavedLocalEligibility::*;

    let mut assignments: IndexVec<GeneratorSavedLocal, SavedLocalEligibility> =
        IndexVec::from_elem(Unassigned, &info.field_tys);

    // The saved locals not eligible for overlap. These will get
    // "promoted" to the prefix of our generator.
    let mut ineligible_locals = BitSet::new_empty(info.field_tys.len());

    // Figure out which of our saved locals are fields in only
    // one variant. The rest are deemed ineligible for overlap.
    for (variant_index, fields) in info.variant_fields.iter_enumerated() {
        for local in fields {
            match assignments[*local] {
                Unassigned => {
                    assignments[*local] = Assigned(variant_index);
                }
                Assigned(idx) => {
                    // We've already seen this local at another suspension
                    // point, so it is no longer a candidate.
                    trace!(
                        "removing local {:?} in >1 variant ({:?}, {:?})",
                        local,
                        variant_index,
                        idx
                    );
                    ineligible_locals.insert(*local);
                    assignments[*local] = Ineligible(None);
                }
                Ineligible(_) => {}
            }
        }
    }

    // Next, check every pair of eligible locals to see if they
    // conflict.
    for local_a in info.storage_conflicts.rows() {
        let conflicts_a = info.storage_conflicts.count(local_a);
        if ineligible_locals.contains(local_a) {
            continue;
        }

        for local_b in info.storage_conflicts.iter(local_a) {
            // local_a and local_b are storage live at the same time, therefore they
            // cannot overlap in the generator layout. The only way to guarantee
            // this is if they are in the same variant, or one is ineligible
            // (which means it is stored in every variant).
            if ineligible_locals.contains(local_b) || assignments[local_a] == assignments[local_b] {
                continue;
            }

            // If they conflict, we will choose one to make ineligible.
            // This is not always optimal; it's just a greedy heuristic that
            // seems to produce good results most of the time.
            let conflicts_b = info.storage_conflicts.count(local_b);
            let (remove, other) =
                if conflicts_a > conflicts_b { (local_a, local_b) } else { (local_b, local_a) };
            ineligible_locals.insert(remove);
            assignments[remove] = Ineligible(None);
            trace!("removing local {:?} due to conflict with {:?}", remove, other);
        }
    }

    // Count the number of variants in use. If only one of them, then it is
    // impossible to overlap any locals in our layout. In this case it's
    // always better to make the remaining locals ineligible, so we can
    // lay them out with the other locals in the prefix and eliminate
    // unnecessary padding bytes.
    {
        let mut used_variants = BitSet::new_empty(info.variant_fields.len());
        for assignment in &assignments {
            if let Assigned(idx) = assignment {
                used_variants.insert(*idx);
            }
        }
        if used_variants.count() < 2 {
            for assignment in assignments.iter_mut() {
                *assignment = Ineligible(None);
            }
            ineligible_locals.insert_all();
        }
    }

    // Write down the order of our locals that will be promoted to the prefix.
    {
        for (idx, local) in ineligible_locals.iter().enumerate() {
            assignments[local] = Ineligible(Some(FieldIdx::from_usize(idx)));
        }
    }
    debug!("generator saved local assignments: {:?}", assignments);

    (ineligible_locals, assignments)
}

/// Compute the full generator layout.
fn generator_layout<'tcx>(
    cx: &LayoutCx<'tcx, TyCtxt<'tcx>>,
    ty: Ty<'tcx>,
    def_id: hir::def_id::DefId,
    args: GenericArgsRef<'tcx>,
) -> Result<Layout<'tcx>, &'tcx LayoutError<'tcx>> {
    use SavedLocalEligibility::*;
    let tcx = cx.tcx;
    let subst_field = |ty: Ty<'tcx>| EarlyBinder::bind(ty).instantiate(tcx, args);

    let Some(info) = tcx.generator_layout(def_id) else {
        return Err(error(cx, LayoutError::Unknown(ty)));
    };
    let (ineligible_locals, assignments) = generator_saved_local_eligibility(&info);

    // Build a prefix layout, including "promoting" all ineligible
    // locals as part of the prefix. We compute the layout of all of
    // these fields at once to get optimal packing.
    let tag_index = args.as_generator().prefix_tys().count();

    // `info.variant_fields` already accounts for the reserved variants, so no need to add them.
    let max_discr = (info.variant_fields.len() - 1) as u128;
    let discr_int = Integer::fit_unsigned(max_discr);
    let tag = Scalar::Initialized {
        value: Primitive::Int(discr_int, false),
        valid_range: WrappingRange { start: 0, end: max_discr },
    };
    let tag_layout = cx.tcx.mk_layout(LayoutS::scalar(cx, tag));

    let promoted_layouts = ineligible_locals
        .iter()
        .map(|local| subst_field(info.field_tys[local].ty))
        .map(|ty| Ty::new_maybe_uninit(tcx, ty))
        .map(|ty| Ok(cx.layout_of(ty)?.layout));
    let prefix_layouts = args
        .as_generator()
        .prefix_tys()
        .map(|ty| Ok(cx.layout_of(ty)?.layout))
        .chain(iter::once(Ok(tag_layout)))
        .chain(promoted_layouts)
        .try_collect::<IndexVec<_, _>>()?;
    let prefix = univariant_uninterned(
        cx,
        ty,
        &prefix_layouts,
        &ReprOptions::default(),
        StructKind::AlwaysSized,
    )?;

    let (prefix_size, prefix_align) = (prefix.size, prefix.align);

    // Split the prefix layout into the "outer" fields (upvars and
    // discriminant) and the "promoted" fields. Promoted fields will
    // get included in each variant that requested them in
    // GeneratorLayout.
    debug!("prefix = {:#?}", prefix);
    let (outer_fields, promoted_offsets, promoted_memory_index) = match prefix.fields {
        FieldsShape::Arbitrary { mut offsets, memory_index } => {
            let mut inverse_memory_index = memory_index.invert_bijective_mapping();

            // "a" (`0..b_start`) and "b" (`b_start..`) correspond to
            // "outer" and "promoted" fields respectively.
            let b_start = FieldIdx::from_usize(tag_index + 1);
            let offsets_b = IndexVec::from_raw(offsets.raw.split_off(b_start.as_usize()));
            let offsets_a = offsets;

            // Disentangle the "a" and "b" components of `inverse_memory_index`
            // by preserving the order but keeping only one disjoint "half" each.
            // FIXME(eddyb) build a better abstraction for permutations, if possible.
            let inverse_memory_index_b: IndexVec<u32, FieldIdx> = inverse_memory_index
                .iter()
                .filter_map(|&i| i.as_u32().checked_sub(b_start.as_u32()).map(FieldIdx::from_u32))
                .collect();
            inverse_memory_index.raw.retain(|&i| i < b_start);
            let inverse_memory_index_a = inverse_memory_index;

            // Since `inverse_memory_index_{a,b}` each only refer to their
            // respective fields, they can be safely inverted
            let memory_index_a = inverse_memory_index_a.invert_bijective_mapping();
            let memory_index_b = inverse_memory_index_b.invert_bijective_mapping();

            let outer_fields =
                FieldsShape::Arbitrary { offsets: offsets_a, memory_index: memory_index_a };
            (outer_fields, offsets_b, memory_index_b)
        }
        _ => bug!(),
    };

    let mut size = prefix.size;
    let mut align = prefix.align;
    let variants = info
        .variant_fields
        .iter_enumerated()
        .map(|(index, variant_fields)| {
            // Only include overlap-eligible fields when we compute our variant layout.
            let variant_only_tys = variant_fields
                .iter()
                .filter(|local| match assignments[**local] {
                    Unassigned => bug!(),
                    Assigned(v) if v == index => true,
                    Assigned(_) => bug!("assignment does not match variant"),
                    Ineligible(_) => false,
                })
                .map(|local| subst_field(info.field_tys[*local].ty));

            let mut variant = univariant_uninterned(
                cx,
                ty,
                &variant_only_tys
                    .map(|ty| Ok(cx.layout_of(ty)?.layout))
                    .try_collect::<IndexVec<_, _>>()?,
                &ReprOptions::default(),
                StructKind::Prefixed(prefix_size, prefix_align.abi),
            )?;
            variant.variants = Variants::Single { index };

            let FieldsShape::Arbitrary { offsets, memory_index } = variant.fields else {
                bug!();
            };

            // Now, stitch the promoted and variant-only fields back together in
            // the order they are mentioned by our GeneratorLayout.
            // Because we only use some subset (that can differ between variants)
            // of the promoted fields, we can't just pick those elements of the
            // `promoted_memory_index` (as we'd end up with gaps).
            // So instead, we build an "inverse memory_index", as if all of the
            // promoted fields were being used, but leave the elements not in the
            // subset as `INVALID_FIELD_IDX`, which we can filter out later to
            // obtain a valid (bijective) mapping.
            const INVALID_FIELD_IDX: FieldIdx = FieldIdx::MAX;
            debug_assert!(variant_fields.next_index() <= INVALID_FIELD_IDX);

            let mut combined_inverse_memory_index = IndexVec::from_elem_n(
                INVALID_FIELD_IDX,
                promoted_memory_index.len() + memory_index.len(),
            );
            let mut offsets_and_memory_index = iter::zip(offsets, memory_index);
            let combined_offsets = variant_fields
                .iter_enumerated()
                .map(|(i, local)| {
                    let (offset, memory_index) = match assignments[*local] {
                        Unassigned => bug!(),
                        Assigned(_) => {
                            let (offset, memory_index) = offsets_and_memory_index.next().unwrap();
                            (offset, promoted_memory_index.len() as u32 + memory_index)
                        }
                        Ineligible(field_idx) => {
                            let field_idx = field_idx.unwrap();
                            (promoted_offsets[field_idx], promoted_memory_index[field_idx])
                        }
                    };
                    combined_inverse_memory_index[memory_index] = i;
                    offset
                })
                .collect();

            // Remove the unused slots and invert the mapping to obtain the
            // combined `memory_index` (also see previous comment).
            combined_inverse_memory_index.raw.retain(|&i| i != INVALID_FIELD_IDX);
            let combined_memory_index = combined_inverse_memory_index.invert_bijective_mapping();

            variant.fields = FieldsShape::Arbitrary {
                offsets: combined_offsets,
                memory_index: combined_memory_index,
            };

            size = size.max(variant.size);
            align = align.max(variant.align);
            Ok(variant)
        })
        .try_collect::<IndexVec<VariantIdx, _>>()?;

    size = size.align_to(align.abi);

    let abi = if prefix.abi.is_uninhabited() || variants.iter().all(|v| v.abi.is_uninhabited()) {
        Abi::Uninhabited
    } else {
        Abi::Aggregate { sized: true }
    };

    let layout = tcx.mk_layout(LayoutS {
        variants: Variants::Multiple {
            tag,
            tag_encoding: TagEncoding::Direct,
            tag_field: tag_index,
            variants,
        },
        fields: outer_fields,
        abi,
        largest_niche: prefix.largest_niche,
        size,
        align,
        max_repr_align: None,
        unadjusted_abi_align: align.abi,
    });
    debug!("generator layout ({:?}): {:#?}", ty, layout);
    Ok(layout)
}

/// This is invoked by the `layout_of` query to record the final
/// layout of each type.
#[inline(always)]
fn record_layout_for_printing<'tcx>(cx: &LayoutCx<'tcx, TyCtxt<'tcx>>, layout: TyAndLayout<'tcx>) {
    // If we are running with `-Zprint-type-sizes`, maybe record layouts
    // for dumping later.
    if cx.tcx.sess.opts.unstable_opts.print_type_sizes {
        record_layout_for_printing_outlined(cx, layout)
    }
}

fn record_layout_for_printing_outlined<'tcx>(
    cx: &LayoutCx<'tcx, TyCtxt<'tcx>>,
    layout: TyAndLayout<'tcx>,
) {
    // Ignore layouts that are done with non-empty environments or
    // non-monomorphic layouts, as the user only wants to see the stuff
    // resulting from the final codegen session.
    if layout.ty.has_non_region_param() || !cx.param_env.caller_bounds().is_empty() {
        return;
    }

    // (delay format until we actually need it)
    let record = |kind, packed, opt_discr_size, variants| {
        let type_desc = format!("{:?}", layout.ty);
        cx.tcx.sess.code_stats.record_type_size(
            kind,
            type_desc,
            layout.align.abi,
            layout.size,
            packed,
            opt_discr_size,
            variants,
        );
    };

    match *layout.ty.kind() {
        ty::Adt(adt_def, _) => {
            debug!("print-type-size t: `{:?}` process adt", layout.ty);
            let adt_kind = adt_def.adt_kind();
            let adt_packed = adt_def.repr().pack.is_some();
            let (variant_infos, opt_discr_size) = variant_info_for_adt(cx, layout, adt_def);
            record(adt_kind.into(), adt_packed, opt_discr_size, variant_infos);
        }

        ty::Generator(def_id, args, _) => {
            debug!("print-type-size t: `{:?}` record generator", layout.ty);
            // Generators always have a begin/poisoned/end state with additional suspend points
            let (variant_infos, opt_discr_size) =
                variant_info_for_generator(cx, layout, def_id, args);
            record(DataTypeKind::Generator, false, opt_discr_size, variant_infos);
        }

        ty::Closure(..) => {
            debug!("print-type-size t: `{:?}` record closure", layout.ty);
            record(DataTypeKind::Closure, false, None, vec![]);
        }

        _ => {
            debug!("print-type-size t: `{:?}` skip non-nominal", layout.ty);
        }
    };
}

fn variant_info_for_adt<'tcx>(
    cx: &LayoutCx<'tcx, TyCtxt<'tcx>>,
    layout: TyAndLayout<'tcx>,
    adt_def: AdtDef<'tcx>,
) -> (Vec<VariantInfo>, Option<Size>) {
    let build_variant_info = |n: Option<Symbol>, flds: &[Symbol], layout: TyAndLayout<'tcx>| {
        let mut min_size = Size::ZERO;
        let field_info: Vec<_> = flds
            .iter()
            .enumerate()
            .map(|(i, &name)| {
                let field_layout = layout.field(cx, i);
                let offset = layout.fields.offset(i);
                min_size = min_size.max(offset + field_layout.size);
                FieldInfo {
                    kind: FieldKind::AdtField,
                    name,
                    offset: offset.bytes(),
                    size: field_layout.size.bytes(),
                    align: field_layout.align.abi.bytes(),
                }
            })
            .collect();

        VariantInfo {
            name: n,
            kind: if layout.is_unsized() { SizeKind::Min } else { SizeKind::Exact },
            align: layout.align.abi.bytes(),
            size: if min_size.bytes() == 0 { layout.size.bytes() } else { min_size.bytes() },
            fields: field_info,
        }
    };

    match layout.variants {
        Variants::Single { index } => {
            if !adt_def.variants().is_empty() && layout.fields != FieldsShape::Primitive {
                debug!("print-type-size `{:#?}` variant {}", layout, adt_def.variant(index).name);
                let variant_def = &adt_def.variant(index);
                let fields: Vec<_> = variant_def.fields.iter().map(|f| f.name).collect();
                (vec![build_variant_info(Some(variant_def.name), &fields, layout)], None)
            } else {
                (vec![], None)
            }
        }

        Variants::Multiple { tag, ref tag_encoding, .. } => {
            debug!(
                "print-type-size `{:#?}` adt general variants def {}",
                layout.ty,
                adt_def.variants().len()
            );
            let variant_infos: Vec<_> = adt_def
                .variants()
                .iter_enumerated()
                .map(|(i, variant_def)| {
                    let fields: Vec<_> = variant_def.fields.iter().map(|f| f.name).collect();
                    build_variant_info(Some(variant_def.name), &fields, layout.for_variant(cx, i))
                })
                .collect();

            (
                variant_infos,
                match tag_encoding {
                    TagEncoding::Direct => Some(tag.size(cx)),
                    _ => None,
                },
            )
        }
    }
}

fn variant_info_for_generator<'tcx>(
    cx: &LayoutCx<'tcx, TyCtxt<'tcx>>,
    layout: TyAndLayout<'tcx>,
    def_id: DefId,
    args: ty::GenericArgsRef<'tcx>,
) -> (Vec<VariantInfo>, Option<Size>) {
    let Variants::Multiple { tag, ref tag_encoding, tag_field, .. } = layout.variants else {
        return (vec![], None);
    };

    let generator = cx.tcx.optimized_mir(def_id).generator_layout().unwrap();
    let upvar_names = cx.tcx.closure_saved_names_of_captured_variables(def_id);

    let mut upvars_size = Size::ZERO;
    let upvar_fields: Vec<_> = args
        .as_generator()
        .upvar_tys()
        .zip(upvar_names)
        .enumerate()
        .map(|(field_idx, (_, name))| {
            let field_layout = layout.field(cx, field_idx);
            let offset = layout.fields.offset(field_idx);
            upvars_size = upvars_size.max(offset + field_layout.size);
            FieldInfo {
                kind: FieldKind::Upvar,
                name: *name,
                offset: offset.bytes(),
                size: field_layout.size.bytes(),
                align: field_layout.align.abi.bytes(),
            }
        })
        .collect();

    let mut variant_infos: Vec<_> = generator
        .variant_fields
        .iter_enumerated()
        .map(|(variant_idx, variant_def)| {
            let variant_layout = layout.for_variant(cx, variant_idx);
            let mut variant_size = Size::ZERO;
            let fields = variant_def
                .iter()
                .enumerate()
                .map(|(field_idx, local)| {
                    let field_layout = variant_layout.field(cx, field_idx);
                    let offset = variant_layout.fields.offset(field_idx);
                    // The struct is as large as the last field's end
                    variant_size = variant_size.max(offset + field_layout.size);
                    FieldInfo {
                        kind: FieldKind::GeneratorLocal,
                        name: generator.field_names[*local].unwrap_or(Symbol::intern(&format!(
                            ".generator_field{}",
                            local.as_usize()
                        ))),
                        offset: offset.bytes(),
                        size: field_layout.size.bytes(),
                        align: field_layout.align.abi.bytes(),
                    }
                })
                .chain(upvar_fields.iter().copied())
                .collect();

            // If the variant has no state-specific fields, then it's the size of the upvars.
            if variant_size == Size::ZERO {
                variant_size = upvars_size;
            }

            // This `if` deserves some explanation.
            //
            // The layout code has a choice of where to place the discriminant of this generator.
            // If the discriminant of the generator is placed early in the layout (before the
            // variant's own fields), then it'll implicitly be counted towards the size of the
            // variant, since we use the maximum offset to calculate size.
            //    (side-note: I know this is a bit problematic given upvars placement, etc).
            //
            // This is important, since the layout printing code always subtracts this discriminant
            // size from the variant size if the struct is "enum"-like, so failing to account for it
            // will either lead to numerical underflow, or an underreported variant size...
            //
            // However, if the discriminant is placed past the end of the variant, then we need
            // to factor in the size of the discriminant manually. This really should be refactored
            // better, but this "works" for now.
            if layout.fields.offset(tag_field) >= variant_size {
                variant_size += match tag_encoding {
                    TagEncoding::Direct => tag.size(cx),
                    _ => Size::ZERO,
                };
            }

            VariantInfo {
                name: Some(Symbol::intern(&ty::GeneratorArgs::variant_name(variant_idx))),
                kind: SizeKind::Exact,
                size: variant_size.bytes(),
                align: variant_layout.align.abi.bytes(),
                fields,
            }
        })
        .collect();

    // The first three variants are hardcoded to be `UNRESUMED`, `RETURNED` and `POISONED`.
    // We will move the `RETURNED` and `POISONED` elements to the end so we
    // are left with a sorting order according to the generators yield points:
    // First `Unresumed`, then the `SuspendN` followed by `Returned` and `Panicked` (POISONED).
    let end_states = variant_infos.drain(1..=2);
    let end_states: Vec<_> = end_states.collect();
    variant_infos.extend(end_states);

    (
        variant_infos,
        match tag_encoding {
            TagEncoding::Direct => Some(tag.size(cx)),
            _ => None,
        },
    )
}
