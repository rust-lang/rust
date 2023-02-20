use rustc_hir as hir;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;
use rustc_middle::mir::{GeneratorLayout, GeneratorSavedLocal};
use rustc_middle::ty::layout::{IntegerExt, LayoutCx, LayoutError, LayoutOf, MAX_SIMD_LANES};
use rustc_middle::ty::{
    self, subst::SubstsRef, EarlyBinder, ReprOptions, Ty, TyCtxt, TypeVisitable,
};
use rustc_span::DUMMY_SP;
use rustc_target::abi::*;

use std::fmt::Debug;
use std::iter;

pub fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers { align_of, ..*providers };
}

#[instrument(skip(tcx, query), level = "debug")]
fn align_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>,
) -> Result<AbiAndPrefAlign, LayoutError<'tcx>> {
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
            return Err(LayoutError::NormalizationFailure(ty, normalization_error));
        }
    };

    if ty != unnormalized_ty {
        // Ensure this layout is also cached for the normalized type.
        return tcx.align_of(param_env.and(ty));
    }

    let cx = LayoutCx { tcx, param_env };

    let alignment = align_of_uncached(&cx, ty)?;

    Ok(alignment)
}

// Invert a bijective mapping, i.e. `invert(map)[y] = x` if `map[x] = y`.
// This is used to go between `memory_index` (source field order to memory order)
// and `inverse_memory_index` (memory order to source field order).
// See also `FieldsShape::Arbitrary::memory_index` for more details.
// FIXME(eddyb) build a better abstraction for permutations, if possible.
fn invert_mapping(map: &[u32]) -> Vec<u32> {
    let mut inverse = vec![0; map.len()];
    for i in 0..map.len() {
        inverse[map[i] as usize] = i as u32;
    }
    inverse
}

fn univariant_uninterned<'tcx>(
    cx: &LayoutCx<'tcx, TyCtxt<'tcx>>,
    ty: Ty<'tcx>,
    fields: &[Layout<'_>],
    repr: &ReprOptions,
    kind: StructKind,
) -> Result<LayoutS, LayoutError<'tcx>> {
    let dl = cx.data_layout();
    let pack = repr.pack;
    if pack.is_some() && repr.align.is_some() {
        cx.tcx.sess.delay_span_bug(DUMMY_SP, "struct cannot be packed and aligned");
        return Err(LayoutError::Unknown(ty));
    }

    cx.univariant(dl, fields, repr, kind).ok_or(LayoutError::SizeOverflow(ty))
}

fn align_of_uncached<'tcx>(
    cx: &LayoutCx<'tcx, TyCtxt<'tcx>>,
    ty: Ty<'tcx>,
) -> Result<AbiAndPrefAlign, LayoutError<'tcx>> {
    let tcx = cx.tcx;
    let param_env = cx.param_env;
    let dl = cx.data_layout();
    let scalar_unit = |value: Primitive| {
        let size = value.size(dl);
        assert!(size.bits() <= 128);
        Scalar::Initialized { value, valid_range: WrappingRange::full(size) }
    };

    let univariant = |fields: &[Layout<'_>], repr: &ReprOptions, kind| {
        Ok(tcx.intern_layout(univariant_uninterned(cx, ty, fields, repr, kind)?))
    };
    debug_assert!(!ty.has_non_region_infer());

    Ok(match *ty.kind() {
        // Basic scalars.
        ty::Bool => I8.align(cx),
        ty::Char => I32.align(cx),
        ty::Int(ity) => Integer::from_int_ty(dl, ity).align(cx),
        ty::Uint(ity) => Integer::from_uint_ty(dl, ity).align(cx),
        ty::Float(fty) => match fty {
            ty::FloatTy::F32 => F32,
            ty::FloatTy::F64 => F64,
        }
        .align(cx),
        ty::FnPtr(_) => Pointer(dl.instruction_address_space).align(cx),

        // The never type.
        ty::Never => tcx.intern_layout(cx.layout_of_never_type()).align(),

        // Potentially-wide pointers.
        ty::Ref(_, pointee, _) | ty::RawPtr(ty::TypeAndMut { ty: pointee, .. }) => {
            let data_ptr_align = Pointer(AddressSpace::DATA).align(cx);

            let pointee = tcx.normalize_erasing_regions(param_env, pointee);
            if pointee.is_sized(tcx, param_env) {
                return Ok(data_ptr_align);
            }

            let unsized_part = tcx.struct_tail_erasing_lifetimes(pointee, param_env);

            let metadata_align = if let Some(metadata_def_id) = tcx.lang_items().metadata_type() {
                let metadata_ty = tcx.normalize_erasing_regions(
                    param_env,
                    tcx.mk_projection(metadata_def_id, [pointee]),
                );
                let metadata_layout = cx.layout_of(metadata_ty)?;
                // If the metadata is a 1-zst, then the pointer is thin.
                if metadata_layout.is_zst() && metadata_layout.align.abi.bytes() == 1 {
                    return Ok(data_ptr_align);
                }

                let Abi::Scalar(metadata) = metadata_layout.abi else {
                    return Err(LayoutError::Unknown(unsized_part));
                };
                metadata.align(dl)
            } else {
                match unsized_part.kind() {
                    ty::Foreign(..) => {
                        return Ok(data_ptr_align);
                    }
                    ty::Slice(_) | ty::Str => dl.ptr_sized_integer().align(cx),
                    ty::Dynamic(..) => data_ptr_align,
                    _ => {
                        return Err(LayoutError::Unknown(unsized_part));
                    }
                }
            };
            // Effectively a (ptr, meta) tuple.
            data_ptr_align.max(metadata_align).max(dl.aggregate_align)
        }

        ty::Dynamic(_, _, ty::DynStar) => {
            let mut data = scalar_unit(Int(dl.ptr_sized_integer(), false));
            data.valid_range_mut().start = 0;
            let mut vtable = scalar_unit(Pointer(AddressSpace::DATA));
            vtable.valid_range_mut().start = 1;
            tcx.intern_layout(cx.scalar_pair(data, vtable)).align()
        }

        // Arrays and slices.
        ty::Array(element, mut count) => {
            if count.has_projections() {
                count = tcx.normalize_erasing_regions(param_env, count);
                if count.has_projections() {
                    return Err(LayoutError::Unknown(ty));
                }
            }
            let element = cx.layout_of(element)?;
            element.align
        }
        ty::Slice(element) => {
            let element = cx.layout_of(element)?;
            element.align
        }

        ty::Str => dl.i8_align,

        // Odd unit types.
        ty::FnDef(..) => univariant(&[], &ReprOptions::default(), StructKind::AlwaysSized)?.align(),
        ty::Dynamic(_, _, ty::Dyn) | ty::Foreign(..) => {
            univariant_uninterned(cx, ty, &[], &ReprOptions::default(), StructKind::AlwaysSized)?
                .align
        }

        ty::Generator(def_id, substs, _) => generator_layout(cx, ty, def_id, substs)?.align(),

        ty::Closure(_, ref substs) => {
            let tys = substs.as_closure().upvar_tys();
            univariant(
                &tys.map(|ty| Ok(cx.layout_of(ty)?.layout)).collect::<Result<Vec<_>, _>>()?,
                &ReprOptions::default(),
                StructKind::AlwaysSized,
            )?
            .align()
        }

        ty::Tuple(tys) => {
            let kind =
                if tys.len() == 0 { StructKind::AlwaysSized } else { StructKind::MaybeUnsized };

            univariant(
                &tys.iter().map(|k| Ok(cx.layout_of(k)?.layout)).collect::<Result<Vec<_>, _>>()?,
                &ReprOptions::default(),
                kind,
            )?
            .align()
        }

        // SIMD vector types.
        ty::Adt(def, substs) if def.repr().simd() => {
            if !def.is_struct() {
                // Should have yielded E0517 by now.
                tcx.sess.delay_span_bug(
                    DUMMY_SP,
                    "#[repr(simd)] was applied to an ADT that is not a struct",
                );
                return Err(LayoutError::Unknown(ty));
            }

            // Supported SIMD vectors are homogeneous ADTs with at least one field:
            //
            // * #[repr(simd)] struct S(T, T, T, T);
            // * #[repr(simd)] struct S { x: T, y: T, z: T, w: T }
            // * #[repr(simd)] struct S([T; 4])
            //
            // where T is a primitive scalar (integer/float/pointer).

            // SIMD vectors with zero fields are not supported.
            // (should be caught by typeck)
            if def.non_enum_variant().fields.is_empty() {
                tcx.sess.fatal(&format!("monomorphising SIMD type `{}` of zero length", ty));
            }

            // Type of the first ADT field:
            let f0_ty = def.non_enum_variant().fields[0].ty(tcx, substs);

            // Heterogeneous SIMD vectors are not supported:
            // (should be caught by typeck)
            for fi in &def.non_enum_variant().fields {
                if fi.ty(tcx, substs) != f0_ty {
                    tcx.sess.fatal(&format!("monomorphising heterogeneous SIMD type `{}`", ty));
                }
            }

            // The element type and number of elements of the SIMD vector
            // are obtained from:
            //
            // * the element type and length of the single array field, if
            // the first field is of array type, or
            //
            // * the homogeneous field type and the number of fields.
            let (e_ty, e_len) = if let ty::Array(e_ty, _) = f0_ty.kind() {
                // First ADT field is an array:

                // SIMD vectors with multiple array fields are not supported:
                // (should be caught by typeck)
                if def.non_enum_variant().fields.len() != 1 {
                    tcx.sess.fatal(&format!(
                        "monomorphising SIMD type `{}` with more than one array field",
                        ty
                    ));
                }

                // Extract the number of elements from the layout of the array field:
                let FieldsShape::Array { count, .. } = cx.layout_of(f0_ty)?.layout.fields() else {
                    return Err(LayoutError::Unknown(ty));
                };

                (*e_ty, *count)
            } else {
                // First ADT field is not an array:
                (f0_ty, def.non_enum_variant().fields.len() as _)
            };

            // SIMD vectors of zero length are not supported.
            // Additionally, lengths are capped at 2^16 as a fixed maximum backends must
            // support.
            //
            // Can't be caught in typeck if the array length is generic.
            if e_len == 0 {
                tcx.sess.fatal(&format!("monomorphising SIMD type `{}` of zero length", ty));
            } else if e_len > MAX_SIMD_LANES {
                tcx.sess.fatal(&format!(
                    "monomorphising SIMD type `{}` of length greater than {}",
                    ty, MAX_SIMD_LANES,
                ));
            }

            // Compute the ABI of the element type:
            let e_ly = cx.layout_of(e_ty)?;
            let Abi::Scalar(_) = e_ly.abi else {
                // This error isn't caught in typeck, e.g., if
                // the element type of the vector is generic.
                tcx.sess.fatal(&format!(
                    "monomorphising SIMD type `{}` with a non-primitive-scalar \
                    (integer/float/pointer) element type `{}`",
                    ty, e_ty
                ))
            };

            // Compute the size and alignment of the vector:
            let size = e_ly.size.checked_mul(e_len, dl).ok_or(LayoutError::SizeOverflow(ty))?;
            dl.vector_align(size)
        }

        // ADTs.
        ty::Adt(def, substs) => {
            // Cache the field layouts.
            let variants = def
                .variants()
                .iter()
                .map(|v| {
                    v.fields
                        .iter()
                        .map(|field| Ok(cx.layout_of(field.ty(tcx, substs))?.layout))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<IndexVec<VariantIdx, _>, _>>()?;

            if def.is_union() {
                if def.repr().pack.is_some() && def.repr().align.is_some() {
                    cx.tcx.sess.delay_span_bug(
                        tcx.def_span(def.did()),
                        "union cannot be packed and aligned",
                    );
                    return Err(LayoutError::Unknown(ty));
                }

                return Ok(tcx
                    .intern_layout(
                        cx.layout_of_union(&def.repr(), &variants)
                            .ok_or(LayoutError::Unknown(ty))?,
                    )
                    .align());
            }
            tcx.intern_layout(
                cx.layout_of_struct_or_enum(
                    &def.repr(),
                    &variants,
                    def.is_enum(),
                    def.is_unsafe_cell(),
                    tcx.layout_scalar_valid_range(def.did()),
                    |min, max| Integer::repr_discr(tcx, ty, &def.repr(), min, max),
                    def.is_enum()
                        .then(|| def.discriminants(tcx).map(|(v, d)| (v, d.val as i128)))
                        .into_iter()
                        .flatten(),
                    def.repr().inhibit_enum_layout_opt()
                        || def
                            .variants()
                            .iter_enumerated()
                            .any(|(i, v)| v.discr != ty::VariantDiscr::Relative(i.as_u32())),
                    {
                        let param_env = tcx.param_env(def.did());
                        def.is_struct()
                            && match def.variants().iter().next().and_then(|x| x.fields.last()) {
                                Some(last_field) => tcx
                                    .type_of(last_field.did)
                                    .subst_identity()
                                    .is_sized(tcx, param_env),
                                None => false,
                            }
                    },
                )
                .ok_or(LayoutError::SizeOverflow(ty))?,
            )
            .align()
        }

        // Types with no meaningful known layout.
        ty::Alias(..) => {
            // NOTE(eddyb) `layout_of` query should've normalized these away,
            // if that was possible, so there's no reason to try again here.
            return Err(LayoutError::Unknown(ty));
        }

        ty::Placeholder(..)
        | ty::GeneratorWitness(..)
        | ty::GeneratorWitnessMIR(..)
        | ty::Infer(_) => {
            bug!("Layout::compute: unexpected type `{}`", ty)
        }

        ty::Bound(..) | ty::Param(_) | ty::Error(_) => {
            return Err(LayoutError::Unknown(ty));
        }
    })
}

/// Overlap eligibility and variant assignment for each GeneratorSavedLocal.
#[derive(Clone, Debug, PartialEq)]
enum SavedLocalEligibility {
    Unassigned,
    Assigned(VariantIdx),
    // FIXME: Use newtype_index so we aren't wasting bytes
    Ineligible(Option<u32>),
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
        IndexVec::from_elem_n(Unassigned, info.field_tys.len());

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
            assignments[local] = Ineligible(Some(idx as u32));
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
    substs: SubstsRef<'tcx>,
) -> Result<Layout<'tcx>, LayoutError<'tcx>> {
    use SavedLocalEligibility::*;
    let tcx = cx.tcx;
    let subst_field = |ty: Ty<'tcx>| EarlyBinder(ty).subst(tcx, substs);

    let Some(info) = tcx.generator_layout(def_id) else {
        return Err(LayoutError::Unknown(ty));
    };
    let (ineligible_locals, assignments) = generator_saved_local_eligibility(&info);

    // Build a prefix layout, including "promoting" all ineligible
    // locals as part of the prefix. We compute the layout of all of
    // these fields at once to get optimal packing.
    let tag_index = substs.as_generator().prefix_tys().count();

    // `info.variant_fields` already accounts for the reserved variants, so no need to add them.
    let max_discr = (info.variant_fields.len() - 1) as u128;
    let discr_int = Integer::fit_unsigned(max_discr);
    let tag = Scalar::Initialized {
        value: Primitive::Int(discr_int, false),
        valid_range: WrappingRange { start: 0, end: max_discr },
    };
    let tag_layout = cx.tcx.intern_layout(LayoutS::scalar(cx, tag));

    let promoted_layouts = ineligible_locals
        .iter()
        .map(|local| subst_field(info.field_tys[local].ty))
        .map(|ty| tcx.mk_maybe_uninit(ty))
        .map(|ty| Ok(cx.layout_of(ty)?.layout));
    let prefix_layouts = substs
        .as_generator()
        .prefix_tys()
        .map(|ty| Ok(cx.layout_of(ty)?.layout))
        .chain(iter::once(Ok(tag_layout)))
        .chain(promoted_layouts)
        .collect::<Result<Vec<_>, _>>()?;
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
            let mut inverse_memory_index = invert_mapping(&memory_index);

            // "a" (`0..b_start`) and "b" (`b_start..`) correspond to
            // "outer" and "promoted" fields respectively.
            let b_start = (tag_index + 1) as u32;
            let offsets_b = offsets.split_off(b_start as usize);
            let offsets_a = offsets;

            // Disentangle the "a" and "b" components of `inverse_memory_index`
            // by preserving the order but keeping only one disjoint "half" each.
            // FIXME(eddyb) build a better abstraction for permutations, if possible.
            let inverse_memory_index_b: Vec<_> =
                inverse_memory_index.iter().filter_map(|&i| i.checked_sub(b_start)).collect();
            inverse_memory_index.retain(|&i| i < b_start);
            let inverse_memory_index_a = inverse_memory_index;

            // Since `inverse_memory_index_{a,b}` each only refer to their
            // respective fields, they can be safely inverted
            let memory_index_a = invert_mapping(&inverse_memory_index_a);
            let memory_index_b = invert_mapping(&inverse_memory_index_b);

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
                    .collect::<Result<Vec<_>, _>>()?,
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
            const INVALID_FIELD_IDX: u32 = !0;
            let mut combined_inverse_memory_index =
                vec![INVALID_FIELD_IDX; promoted_memory_index.len() + memory_index.len()];
            let mut offsets_and_memory_index = iter::zip(offsets, memory_index);
            let combined_offsets = variant_fields
                .iter()
                .enumerate()
                .map(|(i, local)| {
                    let (offset, memory_index) = match assignments[*local] {
                        Unassigned => bug!(),
                        Assigned(_) => {
                            let (offset, memory_index) = offsets_and_memory_index.next().unwrap();
                            (offset, promoted_memory_index.len() as u32 + memory_index)
                        }
                        Ineligible(field_idx) => {
                            let field_idx = field_idx.unwrap() as usize;
                            (promoted_offsets[field_idx], promoted_memory_index[field_idx])
                        }
                    };
                    combined_inverse_memory_index[memory_index as usize] = i as u32;
                    offset
                })
                .collect();

            // Remove the unused slots and invert the mapping to obtain the
            // combined `memory_index` (also see previous comment).
            combined_inverse_memory_index.retain(|&i| i != INVALID_FIELD_IDX);
            let combined_memory_index = invert_mapping(&combined_inverse_memory_index);

            variant.fields = FieldsShape::Arbitrary {
                offsets: combined_offsets,
                memory_index: combined_memory_index,
            };

            size = size.max(variant.size);
            align = align.max(variant.align);
            Ok(variant)
        })
        .collect::<Result<IndexVec<VariantIdx, _>, _>>()?;

    size = size.align_to(align.abi);

    let abi = if prefix.abi.is_uninhabited() || variants.iter().all(|v| v.abi.is_uninhabited()) {
        Abi::Uninhabited
    } else {
        Abi::Aggregate { sized: true }
    };

    let layout = tcx.intern_layout(LayoutS {
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
    });
    debug!("generator layout ({:?}): {:#?}", ty, layout);
    Ok(layout)
}
