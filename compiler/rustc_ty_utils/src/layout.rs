use rustc_hir as hir;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::{Idx, IndexVec};
use rustc_middle::mir::{GeneratorLayout, GeneratorSavedLocal};
use rustc_middle::ty::layout::{
    IntegerExt, LayoutCx, LayoutError, LayoutOf, TyAndLayout, MAX_SIMD_LANES,
};
use rustc_middle::ty::{
    self, subst::SubstsRef, EarlyBinder, ReprOptions, Ty, TyCtxt, TypeVisitable,
};
use rustc_session::{DataTypeKind, FieldInfo, SizeKind, VariantInfo};
use rustc_span::symbol::Symbol;
use rustc_span::DUMMY_SP;
use rustc_target::abi::*;

use std::cmp::{self, Ordering};
use std::iter;
use std::num::NonZeroUsize;
use std::ops::Bound;

use rand::{seq::SliceRandom, SeedableRng};
use rand_xoshiro::Xoshiro128StarStar;

use crate::layout_sanity_check::sanity_check_layout;

pub fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers { layout_of, ..*providers };
}

#[instrument(skip(tcx, query), level = "debug")]
fn layout_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>,
) -> Result<TyAndLayout<'tcx>, LayoutError<'tcx>> {
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
        return tcx.layout_of(param_env.and(ty));
    }

    let cx = LayoutCx { tcx, param_env };

    let layout = layout_of_uncached(&cx, ty)?;
    let layout = TyAndLayout { ty, layout };

    record_layout_for_printing(&cx, layout);

    sanity_check_layout(&cx, &layout);

    Ok(layout)
}

#[derive(Copy, Clone, Debug)]
enum StructKind {
    /// A tuple, closure, or univariant which cannot be coerced to unsized.
    AlwaysSized,
    /// A univariant, the last field of which may be coerced to unsized.
    MaybeUnsized,
    /// A univariant, but with a prefix of an arbitrary size & alignment (e.g., enum tag).
    Prefixed(Size, Align),
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

fn scalar_pair<'tcx>(cx: &LayoutCx<'tcx, TyCtxt<'tcx>>, a: Scalar, b: Scalar) -> LayoutS<'tcx> {
    let dl = cx.data_layout();
    let b_align = b.align(dl);
    let align = a.align(dl).max(b_align).max(dl.aggregate_align);
    let b_offset = a.size(dl).align_to(b_align.abi);
    let size = (b_offset + b.size(dl)).align_to(align.abi);

    // HACK(nox): We iter on `b` and then `a` because `max_by_key`
    // returns the last maximum.
    let largest_niche = Niche::from_scalar(dl, b_offset, b)
        .into_iter()
        .chain(Niche::from_scalar(dl, Size::ZERO, a))
        .max_by_key(|niche| niche.available(dl));

    LayoutS {
        variants: Variants::Single { index: VariantIdx::new(0) },
        fields: FieldsShape::Arbitrary {
            offsets: vec![Size::ZERO, b_offset],
            memory_index: vec![0, 1],
        },
        abi: Abi::ScalarPair(a, b),
        largest_niche,
        align,
        size,
    }
}

fn univariant_uninterned<'tcx>(
    cx: &LayoutCx<'tcx, TyCtxt<'tcx>>,
    ty: Ty<'tcx>,
    fields: &[TyAndLayout<'_>],
    repr: &ReprOptions,
    kind: StructKind,
) -> Result<LayoutS<'tcx>, LayoutError<'tcx>> {
    let dl = cx.data_layout();
    let pack = repr.pack;
    if pack.is_some() && repr.align.is_some() {
        cx.tcx.sess.delay_span_bug(DUMMY_SP, "struct cannot be packed and aligned");
        return Err(LayoutError::Unknown(ty));
    }

    let mut align = if pack.is_some() { dl.i8_align } else { dl.aggregate_align };

    let mut inverse_memory_index: Vec<u32> = (0..fields.len() as u32).collect();

    let optimize = !repr.inhibit_struct_field_reordering_opt();
    if optimize {
        let end = if let StructKind::MaybeUnsized = kind { fields.len() - 1 } else { fields.len() };
        let optimizing = &mut inverse_memory_index[..end];
        let field_align = |f: &TyAndLayout<'_>| {
            if let Some(pack) = pack { f.align.abi.min(pack) } else { f.align.abi }
        };

        // If `-Z randomize-layout` was enabled for the type definition we can shuffle
        // the field ordering to try and catch some code making assumptions about layouts
        // we don't guarantee
        if repr.can_randomize_type_layout() {
            // `ReprOptions.layout_seed` is a deterministic seed that we can use to
            // randomize field ordering with
            let mut rng = Xoshiro128StarStar::seed_from_u64(repr.field_shuffle_seed);

            // Shuffle the ordering of the fields
            optimizing.shuffle(&mut rng);

            // Otherwise we just leave things alone and actually optimize the type's fields
        } else {
            match kind {
                StructKind::AlwaysSized | StructKind::MaybeUnsized => {
                    optimizing.sort_by_key(|&x| {
                        // Place ZSTs first to avoid "interesting offsets",
                        // especially with only one or two non-ZST fields.
                        let f = &fields[x as usize];
                        (!f.is_zst(), cmp::Reverse(field_align(f)))
                    });
                }

                StructKind::Prefixed(..) => {
                    // Sort in ascending alignment so that the layout stays optimal
                    // regardless of the prefix
                    optimizing.sort_by_key(|&x| field_align(&fields[x as usize]));
                }
            }

            // FIXME(Kixiron): We can always shuffle fields within a given alignment class
            //                 regardless of the status of `-Z randomize-layout`
        }
    }

    // inverse_memory_index holds field indices by increasing memory offset.
    // That is, if field 5 has offset 0, the first element of inverse_memory_index is 5.
    // We now write field offsets to the corresponding offset slot;
    // field 5 with offset 0 puts 0 in offsets[5].
    // At the bottom of this function, we invert `inverse_memory_index` to
    // produce `memory_index` (see `invert_mapping`).

    let mut sized = true;
    let mut offsets = vec![Size::ZERO; fields.len()];
    let mut offset = Size::ZERO;
    let mut largest_niche = None;
    let mut largest_niche_available = 0;

    if let StructKind::Prefixed(prefix_size, prefix_align) = kind {
        let prefix_align =
            if let Some(pack) = pack { prefix_align.min(pack) } else { prefix_align };
        align = align.max(AbiAndPrefAlign::new(prefix_align));
        offset = prefix_size.align_to(prefix_align);
    }

    for &i in &inverse_memory_index {
        let field = fields[i as usize];
        if !sized {
            cx.tcx.sess.delay_span_bug(
                DUMMY_SP,
                &format!(
                    "univariant: field #{} of `{}` comes after unsized field",
                    offsets.len(),
                    ty
                ),
            );
        }

        if field.is_unsized() {
            sized = false;
        }

        // Invariant: offset < dl.obj_size_bound() <= 1<<61
        let field_align = if let Some(pack) = pack {
            field.align.min(AbiAndPrefAlign::new(pack))
        } else {
            field.align
        };
        offset = offset.align_to(field_align.abi);
        align = align.max(field_align);

        debug!("univariant offset: {:?} field: {:#?}", offset, field);
        offsets[i as usize] = offset;

        if let Some(mut niche) = field.largest_niche {
            let available = niche.available(dl);
            if available > largest_niche_available {
                largest_niche_available = available;
                niche.offset += offset;
                largest_niche = Some(niche);
            }
        }

        offset = offset.checked_add(field.size, dl).ok_or(LayoutError::SizeOverflow(ty))?;
    }

    if let Some(repr_align) = repr.align {
        align = align.max(AbiAndPrefAlign::new(repr_align));
    }

    debug!("univariant min_size: {:?}", offset);
    let min_size = offset;

    // As stated above, inverse_memory_index holds field indices by increasing offset.
    // This makes it an already-sorted view of the offsets vec.
    // To invert it, consider:
    // If field 5 has offset 0, offsets[0] is 5, and memory_index[5] should be 0.
    // Field 5 would be the first element, so memory_index is i:
    // Note: if we didn't optimize, it's already right.

    let memory_index =
        if optimize { invert_mapping(&inverse_memory_index) } else { inverse_memory_index };

    let size = min_size.align_to(align.abi);
    let mut abi = Abi::Aggregate { sized };

    // Unpack newtype ABIs and find scalar pairs.
    if sized && size.bytes() > 0 {
        // All other fields must be ZSTs.
        let mut non_zst_fields = fields.iter().enumerate().filter(|&(_, f)| !f.is_zst());

        match (non_zst_fields.next(), non_zst_fields.next(), non_zst_fields.next()) {
            // We have exactly one non-ZST field.
            (Some((i, field)), None, None) => {
                // Field fills the struct and it has a scalar or scalar pair ABI.
                if offsets[i].bytes() == 0 && align.abi == field.align.abi && size == field.size {
                    match field.abi {
                        // For plain scalars, or vectors of them, we can't unpack
                        // newtypes for `#[repr(C)]`, as that affects C ABIs.
                        Abi::Scalar(_) | Abi::Vector { .. } if optimize => {
                            abi = field.abi;
                        }
                        // But scalar pairs are Rust-specific and get
                        // treated as aggregates by C ABIs anyway.
                        Abi::ScalarPair(..) => {
                            abi = field.abi;
                        }
                        _ => {}
                    }
                }
            }

            // Two non-ZST fields, and they're both scalars.
            (Some((i, a)), Some((j, b)), None) => {
                match (a.abi, b.abi) {
                    (Abi::Scalar(a), Abi::Scalar(b)) => {
                        // Order by the memory placement, not source order.
                        let ((i, a), (j, b)) = if offsets[i] < offsets[j] {
                            ((i, a), (j, b))
                        } else {
                            ((j, b), (i, a))
                        };
                        let pair = scalar_pair(cx, a, b);
                        let pair_offsets = match pair.fields {
                            FieldsShape::Arbitrary { ref offsets, ref memory_index } => {
                                assert_eq!(memory_index, &[0, 1]);
                                offsets
                            }
                            _ => bug!(),
                        };
                        if offsets[i] == pair_offsets[0]
                            && offsets[j] == pair_offsets[1]
                            && align == pair.align
                            && size == pair.size
                        {
                            // We can use `ScalarPair` only when it matches our
                            // already computed layout (including `#[repr(C)]`).
                            abi = pair.abi;
                        }
                    }
                    _ => {}
                }
            }

            _ => {}
        }
    }

    if fields.iter().any(|f| f.abi.is_uninhabited()) {
        abi = Abi::Uninhabited;
    }

    Ok(LayoutS {
        variants: Variants::Single { index: VariantIdx::new(0) },
        fields: FieldsShape::Arbitrary { offsets, memory_index },
        abi,
        largest_niche,
        align,
        size,
    })
}

fn layout_of_uncached<'tcx>(
    cx: &LayoutCx<'tcx, TyCtxt<'tcx>>,
    ty: Ty<'tcx>,
) -> Result<Layout<'tcx>, LayoutError<'tcx>> {
    let tcx = cx.tcx;
    let param_env = cx.param_env;
    let dl = cx.data_layout();
    let scalar_unit = |value: Primitive| {
        let size = value.size(dl);
        assert!(size.bits() <= 128);
        Scalar::Initialized { value, valid_range: WrappingRange::full(size) }
    };
    let scalar = |value: Primitive| tcx.intern_layout(LayoutS::scalar(cx, scalar_unit(value)));

    let univariant = |fields: &[TyAndLayout<'_>], repr: &ReprOptions, kind| {
        Ok(tcx.intern_layout(univariant_uninterned(cx, ty, fields, repr, kind)?))
    };
    debug_assert!(!ty.has_non_region_infer());

    Ok(match *ty.kind() {
        // Basic scalars.
        ty::Bool => tcx.intern_layout(LayoutS::scalar(
            cx,
            Scalar::Initialized {
                value: Int(I8, false),
                valid_range: WrappingRange { start: 0, end: 1 },
            },
        )),
        ty::Char => tcx.intern_layout(LayoutS::scalar(
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
            let mut ptr = scalar_unit(Pointer);
            ptr.valid_range_mut().start = 1;
            tcx.intern_layout(LayoutS::scalar(cx, ptr))
        }

        // The never type.
        ty::Never => tcx.intern_layout(LayoutS {
            variants: Variants::Single { index: VariantIdx::new(0) },
            fields: FieldsShape::Primitive,
            abi: Abi::Uninhabited,
            largest_niche: None,
            align: dl.i8_align,
            size: Size::ZERO,
        }),

        // Potentially-wide pointers.
        ty::Ref(_, pointee, _) | ty::RawPtr(ty::TypeAndMut { ty: pointee, .. }) => {
            let mut data_ptr = scalar_unit(Pointer);
            if !ty.is_unsafe_ptr() {
                data_ptr.valid_range_mut().start = 1;
            }

            let pointee = tcx.normalize_erasing_regions(param_env, pointee);
            if pointee.is_sized(tcx, param_env) {
                return Ok(tcx.intern_layout(LayoutS::scalar(cx, data_ptr)));
            }

            let unsized_part = tcx.struct_tail_erasing_lifetimes(pointee, param_env);
            let metadata = match unsized_part.kind() {
                ty::Foreign(..) => {
                    return Ok(tcx.intern_layout(LayoutS::scalar(cx, data_ptr)));
                }
                ty::Slice(_) | ty::Str => scalar_unit(Int(dl.ptr_sized_integer(), false)),
                ty::Dynamic(..) => {
                    let mut vtable = scalar_unit(Pointer);
                    vtable.valid_range_mut().start = 1;
                    vtable
                }
                _ => return Err(LayoutError::Unknown(unsized_part)),
            };

            // Effectively a (ptr, meta) tuple.
            tcx.intern_layout(scalar_pair(cx, data_ptr, metadata))
        }

        ty::Dynamic(_, _, ty::DynStar) => {
            let mut data = scalar_unit(Int(dl.ptr_sized_integer(), false));
            data.valid_range_mut().start = 0;
            let mut vtable = scalar_unit(Pointer);
            vtable.valid_range_mut().start = 1;
            tcx.intern_layout(scalar_pair(cx, data, vtable))
        }

        // Arrays and slices.
        ty::Array(element, mut count) => {
            if count.has_projections() {
                count = tcx.normalize_erasing_regions(param_env, count);
                if count.has_projections() {
                    return Err(LayoutError::Unknown(ty));
                }
            }

            let count = count.try_eval_usize(tcx, param_env).ok_or(LayoutError::Unknown(ty))?;
            let element = cx.layout_of(element)?;
            let size = element.size.checked_mul(count, dl).ok_or(LayoutError::SizeOverflow(ty))?;

            let abi = if count != 0 && tcx.conservative_is_privately_uninhabited(param_env.and(ty))
            {
                Abi::Uninhabited
            } else {
                Abi::Aggregate { sized: true }
            };

            let largest_niche = if count != 0 { element.largest_niche } else { None };

            tcx.intern_layout(LayoutS {
                variants: Variants::Single { index: VariantIdx::new(0) },
                fields: FieldsShape::Array { stride: element.size, count },
                abi,
                largest_niche,
                align: element.align,
                size,
            })
        }
        ty::Slice(element) => {
            let element = cx.layout_of(element)?;
            tcx.intern_layout(LayoutS {
                variants: Variants::Single { index: VariantIdx::new(0) },
                fields: FieldsShape::Array { stride: element.size, count: 0 },
                abi: Abi::Aggregate { sized: false },
                largest_niche: None,
                align: element.align,
                size: Size::ZERO,
            })
        }
        ty::Str => tcx.intern_layout(LayoutS {
            variants: Variants::Single { index: VariantIdx::new(0) },
            fields: FieldsShape::Array { stride: Size::from_bytes(1), count: 0 },
            abi: Abi::Aggregate { sized: false },
            largest_niche: None,
            align: dl.i8_align,
            size: Size::ZERO,
        }),

        // Odd unit types.
        ty::FnDef(..) => univariant(&[], &ReprOptions::default(), StructKind::AlwaysSized)?,
        ty::Dynamic(_, _, ty::Dyn) | ty::Foreign(..) => {
            let mut unit = univariant_uninterned(
                cx,
                ty,
                &[],
                &ReprOptions::default(),
                StructKind::AlwaysSized,
            )?;
            match unit.abi {
                Abi::Aggregate { ref mut sized } => *sized = false,
                _ => bug!(),
            }
            tcx.intern_layout(unit)
        }

        ty::Generator(def_id, substs, _) => generator_layout(cx, ty, def_id, substs)?,

        ty::Closure(_, ref substs) => {
            let tys = substs.as_closure().upvar_tys();
            univariant(
                &tys.map(|ty| cx.layout_of(ty)).collect::<Result<Vec<_>, _>>()?,
                &ReprOptions::default(),
                StructKind::AlwaysSized,
            )?
        }

        ty::Tuple(tys) => {
            let kind =
                if tys.len() == 0 { StructKind::AlwaysSized } else { StructKind::MaybeUnsized };

            univariant(
                &tys.iter().map(|k| cx.layout_of(k)).collect::<Result<Vec<_>, _>>()?,
                &ReprOptions::default(),
                kind,
            )?
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
            let (e_ty, e_len, is_array) = if let ty::Array(e_ty, _) = f0_ty.kind() {
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
                tcx.sess.fatal(&format!("monomorphising SIMD type `{}` of zero length", ty));
            } else if e_len > MAX_SIMD_LANES {
                tcx.sess.fatal(&format!(
                    "monomorphising SIMD type `{}` of length greater than {}",
                    ty, MAX_SIMD_LANES,
                ));
            }

            // Compute the ABI of the element type:
            let e_ly = cx.layout_of(e_ty)?;
            let Abi::Scalar(e_abi) = e_ly.abi else {
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
            let align = dl.vector_align(size);
            let size = size.align_to(align.abi);

            // Compute the placement of the vector fields:
            let fields = if is_array {
                FieldsShape::Arbitrary { offsets: vec![Size::ZERO], memory_index: vec![0] }
            } else {
                FieldsShape::Array { stride: e_ly.size, count: e_len }
            };

            tcx.intern_layout(LayoutS {
                variants: Variants::Single { index: VariantIdx::new(0) },
                fields,
                abi: Abi::Vector { element: e_abi, count: e_len },
                largest_niche: e_ly.largest_niche,
                size,
                align,
            })
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
                        .map(|field| cx.layout_of(field.ty(tcx, substs)))
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

                let mut align =
                    if def.repr().pack.is_some() { dl.i8_align } else { dl.aggregate_align };

                if let Some(repr_align) = def.repr().align {
                    align = align.max(AbiAndPrefAlign::new(repr_align));
                }

                let optimize = !def.repr().inhibit_union_abi_opt();
                let mut size = Size::ZERO;
                let mut abi = Abi::Aggregate { sized: true };
                let index = VariantIdx::new(0);
                for field in &variants[index] {
                    assert!(!field.is_unsized());
                    align = align.max(field.align);

                    // If all non-ZST fields have the same ABI, forward this ABI
                    if optimize && !field.is_zst() {
                        // Discard valid range information and allow undef
                        let field_abi = match field.abi {
                            Abi::Scalar(x) => Abi::Scalar(x.to_union()),
                            Abi::ScalarPair(x, y) => Abi::ScalarPair(x.to_union(), y.to_union()),
                            Abi::Vector { element: x, count } => {
                                Abi::Vector { element: x.to_union(), count }
                            }
                            Abi::Uninhabited | Abi::Aggregate { .. } => {
                                Abi::Aggregate { sized: true }
                            }
                        };

                        if size == Size::ZERO {
                            // first non ZST: initialize 'abi'
                            abi = field_abi;
                        } else if abi != field_abi {
                            // different fields have different ABI: reset to Aggregate
                            abi = Abi::Aggregate { sized: true };
                        }
                    }

                    size = cmp::max(size, field.size);
                }

                if let Some(pack) = def.repr().pack {
                    align = align.min(AbiAndPrefAlign::new(pack));
                }

                return Ok(tcx.intern_layout(LayoutS {
                    variants: Variants::Single { index },
                    fields: FieldsShape::Union(
                        NonZeroUsize::new(variants[index].len()).ok_or(LayoutError::Unknown(ty))?,
                    ),
                    abi,
                    largest_niche: None,
                    align,
                    size: size.align_to(align.abi),
                }));
            }

            // A variant is absent if it's uninhabited and only has ZST fields.
            // Present uninhabited variants only require space for their fields,
            // but *not* an encoding of the discriminant (e.g., a tag value).
            // See issue #49298 for more details on the need to leave space
            // for non-ZST uninhabited data (mostly partial initialization).
            let absent = |fields: &[TyAndLayout<'_>]| {
                let uninhabited = fields.iter().any(|f| f.abi.is_uninhabited());
                let is_zst = fields.iter().all(|f| f.is_zst());
                uninhabited && is_zst
            };
            let (present_first, present_second) = {
                let mut present_variants = variants
                    .iter_enumerated()
                    .filter_map(|(i, v)| if absent(v) { None } else { Some(i) });
                (present_variants.next(), present_variants.next())
            };
            let present_first = match present_first {
                Some(present_first) => present_first,
                // Uninhabited because it has no variants, or only absent ones.
                None if def.is_enum() => {
                    return Ok(tcx.layout_of(param_env.and(tcx.types.never))?.layout);
                }
                // If it's a struct, still compute a layout so that we can still compute the
                // field offsets.
                None => VariantIdx::new(0),
            };

            let is_struct = !def.is_enum() ||
                    // Only one variant is present.
                    (present_second.is_none() &&
                        // Representation optimizations are allowed.
                        !def.repr().inhibit_enum_layout_opt());
            if is_struct {
                // Struct, or univariant enum equivalent to a struct.
                // (Typechecking will reject discriminant-sizing attrs.)

                let v = present_first;
                let kind = if def.is_enum() || variants[v].is_empty() {
                    StructKind::AlwaysSized
                } else {
                    let param_env = tcx.param_env(def.did());
                    let last_field = def.variant(v).fields.last().unwrap();
                    let always_sized = tcx.type_of(last_field.did).is_sized(tcx, param_env);
                    if !always_sized { StructKind::MaybeUnsized } else { StructKind::AlwaysSized }
                };

                let mut st = univariant_uninterned(cx, ty, &variants[v], &def.repr(), kind)?;
                st.variants = Variants::Single { index: v };

                if def.is_unsafe_cell() {
                    let hide_niches = |scalar: &mut _| match scalar {
                        Scalar::Initialized { value, valid_range } => {
                            *valid_range = WrappingRange::full(value.size(dl))
                        }
                        // Already doesn't have any niches
                        Scalar::Union { .. } => {}
                    };
                    match &mut st.abi {
                        Abi::Uninhabited => {}
                        Abi::Scalar(scalar) => hide_niches(scalar),
                        Abi::ScalarPair(a, b) => {
                            hide_niches(a);
                            hide_niches(b);
                        }
                        Abi::Vector { element, count: _ } => hide_niches(element),
                        Abi::Aggregate { sized: _ } => {}
                    }
                    st.largest_niche = None;
                    return Ok(tcx.intern_layout(st));
                }

                let (start, end) = cx.tcx.layout_scalar_valid_range(def.did());
                match st.abi {
                    Abi::Scalar(ref mut scalar) | Abi::ScalarPair(ref mut scalar, _) => {
                        // the asserts ensure that we are not using the
                        // `#[rustc_layout_scalar_valid_range(n)]`
                        // attribute to widen the range of anything as that would probably
                        // result in UB somewhere
                        // FIXME(eddyb) the asserts are probably not needed,
                        // as larger validity ranges would result in missed
                        // optimizations, *not* wrongly assuming the inner
                        // value is valid. e.g. unions enlarge validity ranges,
                        // because the values may be uninitialized.
                        if let Bound::Included(start) = start {
                            // FIXME(eddyb) this might be incorrect - it doesn't
                            // account for wrap-around (end < start) ranges.
                            let valid_range = scalar.valid_range_mut();
                            assert!(valid_range.start <= start);
                            valid_range.start = start;
                        }
                        if let Bound::Included(end) = end {
                            // FIXME(eddyb) this might be incorrect - it doesn't
                            // account for wrap-around (end < start) ranges.
                            let valid_range = scalar.valid_range_mut();
                            assert!(valid_range.end >= end);
                            valid_range.end = end;
                        }

                        // Update `largest_niche` if we have introduced a larger niche.
                        let niche = Niche::from_scalar(dl, Size::ZERO, *scalar);
                        if let Some(niche) = niche {
                            match st.largest_niche {
                                Some(largest_niche) => {
                                    // Replace the existing niche even if they're equal,
                                    // because this one is at a lower offset.
                                    if largest_niche.available(dl) <= niche.available(dl) {
                                        st.largest_niche = Some(niche);
                                    }
                                }
                                None => st.largest_niche = Some(niche),
                            }
                        }
                    }
                    _ => assert!(
                        start == Bound::Unbounded && end == Bound::Unbounded,
                        "nonscalar layout for layout_scalar_valid_range type {:?}: {:#?}",
                        def,
                        st,
                    ),
                }

                return Ok(tcx.intern_layout(st));
            }

            // At this point, we have handled all unions and
            // structs. (We have also handled univariant enums
            // that allow representation optimization.)
            assert!(def.is_enum());

            // Until we've decided whether to use the tagged or
            // niche filling LayoutS, we don't want to intern the
            // variant layouts, so we can't store them in the
            // overall LayoutS. Store the overall LayoutS
            // and the variant LayoutSs here until then.
            struct TmpLayout<'tcx> {
                layout: LayoutS<'tcx>,
                variants: IndexVec<VariantIdx, LayoutS<'tcx>>,
            }

            let calculate_niche_filling_layout =
                || -> Result<Option<TmpLayout<'tcx>>, LayoutError<'tcx>> {
                    // The current code for niche-filling relies on variant indices
                    // instead of actual discriminants, so enums with
                    // explicit discriminants (RFC #2363) would misbehave.
                    if def.repr().inhibit_enum_layout_opt()
                        || def
                            .variants()
                            .iter_enumerated()
                            .any(|(i, v)| v.discr != ty::VariantDiscr::Relative(i.as_u32()))
                    {
                        return Ok(None);
                    }

                    if variants.len() < 2 {
                        return Ok(None);
                    }

                    let mut align = dl.aggregate_align;
                    let mut variant_layouts = variants
                        .iter_enumerated()
                        .map(|(j, v)| {
                            let mut st = univariant_uninterned(
                                cx,
                                ty,
                                v,
                                &def.repr(),
                                StructKind::AlwaysSized,
                            )?;
                            st.variants = Variants::Single { index: j };

                            align = align.max(st.align);

                            Ok(st)
                        })
                        .collect::<Result<IndexVec<VariantIdx, _>, _>>()?;

                    let largest_variant_index = match variant_layouts
                        .iter_enumerated()
                        .max_by_key(|(_i, layout)| layout.size.bytes())
                        .map(|(i, _layout)| i)
                    {
                        None => return Ok(None),
                        Some(i) => i,
                    };

                    let all_indices = VariantIdx::new(0)..=VariantIdx::new(variants.len() - 1);
                    let needs_disc = |index: VariantIdx| {
                        index != largest_variant_index && !absent(&variants[index])
                    };
                    let niche_variants = all_indices.clone().find(|v| needs_disc(*v)).unwrap()
                        ..=all_indices.rev().find(|v| needs_disc(*v)).unwrap();

                    let count = niche_variants.size_hint().1.unwrap() as u128;

                    // Find the field with the largest niche
                    let (field_index, niche, (niche_start, niche_scalar)) = match variants
                        [largest_variant_index]
                        .iter()
                        .enumerate()
                        .filter_map(|(j, field)| Some((j, field.largest_niche?)))
                        .max_by_key(|(_, niche)| niche.available(dl))
                        .and_then(|(j, niche)| Some((j, niche, niche.reserve(cx, count)?)))
                    {
                        None => return Ok(None),
                        Some(x) => x,
                    };

                    let niche_offset = niche.offset
                        + variant_layouts[largest_variant_index].fields.offset(field_index);
                    let niche_size = niche.value.size(dl);
                    let size = variant_layouts[largest_variant_index].size.align_to(align.abi);

                    let all_variants_fit =
                        variant_layouts.iter_enumerated_mut().all(|(i, layout)| {
                            if i == largest_variant_index {
                                return true;
                            }

                            layout.largest_niche = None;

                            if layout.size <= niche_offset {
                                // This variant will fit before the niche.
                                return true;
                            }

                            // Determine if it'll fit after the niche.
                            let this_align = layout.align.abi;
                            let this_offset = (niche_offset + niche_size).align_to(this_align);

                            if this_offset + layout.size > size {
                                return false;
                            }

                            // It'll fit, but we need to make some adjustments.
                            match layout.fields {
                                FieldsShape::Arbitrary { ref mut offsets, .. } => {
                                    for (j, offset) in offsets.iter_mut().enumerate() {
                                        if !variants[i][j].is_zst() {
                                            *offset += this_offset;
                                        }
                                    }
                                }
                                _ => {
                                    panic!("Layout of fields should be Arbitrary for variants")
                                }
                            }

                            // It can't be a Scalar or ScalarPair because the offset isn't 0.
                            if !layout.abi.is_uninhabited() {
                                layout.abi = Abi::Aggregate { sized: true };
                            }
                            layout.size += this_offset;

                            true
                        });

                    if !all_variants_fit {
                        return Ok(None);
                    }

                    let largest_niche = Niche::from_scalar(dl, niche_offset, niche_scalar);

                    let others_zst = variant_layouts
                        .iter_enumerated()
                        .all(|(i, layout)| i == largest_variant_index || layout.size == Size::ZERO);
                    let same_size = size == variant_layouts[largest_variant_index].size;
                    let same_align = align == variant_layouts[largest_variant_index].align;

                    let abi = if variant_layouts.iter().all(|v| v.abi.is_uninhabited()) {
                        Abi::Uninhabited
                    } else if same_size && same_align && others_zst {
                        match variant_layouts[largest_variant_index].abi {
                            // When the total alignment and size match, we can use the
                            // same ABI as the scalar variant with the reserved niche.
                            Abi::Scalar(_) => Abi::Scalar(niche_scalar),
                            Abi::ScalarPair(first, second) => {
                                // Only the niche is guaranteed to be initialised,
                                // so use union layouts for the other primitive.
                                if niche_offset == Size::ZERO {
                                    Abi::ScalarPair(niche_scalar, second.to_union())
                                } else {
                                    Abi::ScalarPair(first.to_union(), niche_scalar)
                                }
                            }
                            _ => Abi::Aggregate { sized: true },
                        }
                    } else {
                        Abi::Aggregate { sized: true }
                    };

                    let layout = LayoutS {
                        variants: Variants::Multiple {
                            tag: niche_scalar,
                            tag_encoding: TagEncoding::Niche {
                                untagged_variant: largest_variant_index,
                                niche_variants,
                                niche_start,
                            },
                            tag_field: 0,
                            variants: IndexVec::new(),
                        },
                        fields: FieldsShape::Arbitrary {
                            offsets: vec![niche_offset],
                            memory_index: vec![0],
                        },
                        abi,
                        largest_niche,
                        size,
                        align,
                    };

                    Ok(Some(TmpLayout { layout, variants: variant_layouts }))
                };

            let niche_filling_layout = calculate_niche_filling_layout()?;

            let (mut min, mut max) = (i128::MAX, i128::MIN);
            let discr_type = def.repr().discr_type();
            let bits = Integer::from_attr(cx, discr_type).size().bits();
            for (i, discr) in def.discriminants(tcx) {
                if variants[i].iter().any(|f| f.abi.is_uninhabited()) {
                    continue;
                }
                let mut x = discr.val as i128;
                if discr_type.is_signed() {
                    // sign extend the raw representation to be an i128
                    x = (x << (128 - bits)) >> (128 - bits);
                }
                if x < min {
                    min = x;
                }
                if x > max {
                    max = x;
                }
            }
            // We might have no inhabited variants, so pretend there's at least one.
            if (min, max) == (i128::MAX, i128::MIN) {
                min = 0;
                max = 0;
            }
            assert!(min <= max, "discriminant range is {}...{}", min, max);
            let (min_ity, signed) = Integer::repr_discr(tcx, ty, &def.repr(), min, max);

            let mut align = dl.aggregate_align;
            let mut size = Size::ZERO;

            // We're interested in the smallest alignment, so start large.
            let mut start_align = Align::from_bytes(256).unwrap();
            assert_eq!(Integer::for_align(dl, start_align), None);

            // repr(C) on an enum tells us to make a (tag, union) layout,
            // so we need to grow the prefix alignment to be at least
            // the alignment of the union. (This value is used both for
            // determining the alignment of the overall enum, and the
            // determining the alignment of the payload after the tag.)
            let mut prefix_align = min_ity.align(dl).abi;
            if def.repr().c() {
                for fields in &variants {
                    for field in fields {
                        prefix_align = prefix_align.max(field.align.abi);
                    }
                }
            }

            // Create the set of structs that represent each variant.
            let mut layout_variants = variants
                .iter_enumerated()
                .map(|(i, field_layouts)| {
                    let mut st = univariant_uninterned(
                        cx,
                        ty,
                        &field_layouts,
                        &def.repr(),
                        StructKind::Prefixed(min_ity.size(), prefix_align),
                    )?;
                    st.variants = Variants::Single { index: i };
                    // Find the first field we can't move later
                    // to make room for a larger discriminant.
                    for field in st.fields.index_by_increasing_offset().map(|j| field_layouts[j]) {
                        if !field.is_zst() || field.align.abi.bytes() != 1 {
                            start_align = start_align.min(field.align.abi);
                            break;
                        }
                    }
                    size = cmp::max(size, st.size);
                    align = align.max(st.align);
                    Ok(st)
                })
                .collect::<Result<IndexVec<VariantIdx, _>, _>>()?;

            // Align the maximum variant size to the largest alignment.
            size = size.align_to(align.abi);

            if size.bytes() >= dl.obj_size_bound() {
                return Err(LayoutError::SizeOverflow(ty));
            }

            let typeck_ity = Integer::from_attr(dl, def.repr().discr_type());
            if typeck_ity < min_ity {
                // It is a bug if Layout decided on a greater discriminant size than typeck for
                // some reason at this point (based on values discriminant can take on). Mostly
                // because this discriminant will be loaded, and then stored into variable of
                // type calculated by typeck. Consider such case (a bug): typeck decided on
                // byte-sized discriminant, but layout thinks we need a 16-bit to store all
                // discriminant values. That would be a bug, because then, in codegen, in order
                // to store this 16-bit discriminant into 8-bit sized temporary some of the
                // space necessary to represent would have to be discarded (or layout is wrong
                // on thinking it needs 16 bits)
                bug!(
                    "layout decided on a larger discriminant type ({:?}) than typeck ({:?})",
                    min_ity,
                    typeck_ity
                );
                // However, it is fine to make discr type however large (as an optimisation)
                // after this point – we’ll just truncate the value we load in codegen.
            }

            // Check to see if we should use a different type for the
            // discriminant. We can safely use a type with the same size
            // as the alignment of the first field of each variant.
            // We increase the size of the discriminant to avoid LLVM copying
            // padding when it doesn't need to. This normally causes unaligned
            // load/stores and excessive memcpy/memset operations. By using a
            // bigger integer size, LLVM can be sure about its contents and
            // won't be so conservative.

            // Use the initial field alignment
            let mut ity = if def.repr().c() || def.repr().int.is_some() {
                min_ity
            } else {
                Integer::for_align(dl, start_align).unwrap_or(min_ity)
            };

            // If the alignment is not larger than the chosen discriminant size,
            // don't use the alignment as the final size.
            if ity <= min_ity {
                ity = min_ity;
            } else {
                // Patch up the variants' first few fields.
                let old_ity_size = min_ity.size();
                let new_ity_size = ity.size();
                for variant in &mut layout_variants {
                    match variant.fields {
                        FieldsShape::Arbitrary { ref mut offsets, .. } => {
                            for i in offsets {
                                if *i <= old_ity_size {
                                    assert_eq!(*i, old_ity_size);
                                    *i = new_ity_size;
                                }
                            }
                            // We might be making the struct larger.
                            if variant.size <= old_ity_size {
                                variant.size = new_ity_size;
                            }
                        }
                        _ => bug!(),
                    }
                }
            }

            let tag_mask = ity.size().unsigned_int_max();
            let tag = Scalar::Initialized {
                value: Int(ity, signed),
                valid_range: WrappingRange {
                    start: (min as u128 & tag_mask),
                    end: (max as u128 & tag_mask),
                },
            };
            let mut abi = Abi::Aggregate { sized: true };

            if layout_variants.iter().all(|v| v.abi.is_uninhabited()) {
                abi = Abi::Uninhabited;
            } else if tag.size(dl) == size {
                // Make sure we only use scalar layout when the enum is entirely its
                // own tag (i.e. it has no padding nor any non-ZST variant fields).
                abi = Abi::Scalar(tag);
            } else {
                // Try to use a ScalarPair for all tagged enums.
                let mut common_prim = None;
                let mut common_prim_initialized_in_all_variants = true;
                for (field_layouts, layout_variant) in iter::zip(&variants, &layout_variants) {
                    let FieldsShape::Arbitrary { ref offsets, .. } = layout_variant.fields else {
                            bug!();
                        };
                    let mut fields = iter::zip(field_layouts, offsets).filter(|p| !p.0.is_zst());
                    let (field, offset) = match (fields.next(), fields.next()) {
                        (None, None) => {
                            common_prim_initialized_in_all_variants = false;
                            continue;
                        }
                        (Some(pair), None) => pair,
                        _ => {
                            common_prim = None;
                            break;
                        }
                    };
                    let prim = match field.abi {
                        Abi::Scalar(scalar) => {
                            common_prim_initialized_in_all_variants &=
                                matches!(scalar, Scalar::Initialized { .. });
                            scalar.primitive()
                        }
                        _ => {
                            common_prim = None;
                            break;
                        }
                    };
                    if let Some(pair) = common_prim {
                        // This is pretty conservative. We could go fancier
                        // by conflating things like i32 and u32, or even
                        // realising that (u8, u8) could just cohabit with
                        // u16 or even u32.
                        if pair != (prim, offset) {
                            common_prim = None;
                            break;
                        }
                    } else {
                        common_prim = Some((prim, offset));
                    }
                }
                if let Some((prim, offset)) = common_prim {
                    let prim_scalar = if common_prim_initialized_in_all_variants {
                        scalar_unit(prim)
                    } else {
                        // Common prim might be uninit.
                        Scalar::Union { value: prim }
                    };
                    let pair = scalar_pair(cx, tag, prim_scalar);
                    let pair_offsets = match pair.fields {
                        FieldsShape::Arbitrary { ref offsets, ref memory_index } => {
                            assert_eq!(memory_index, &[0, 1]);
                            offsets
                        }
                        _ => bug!(),
                    };
                    if pair_offsets[0] == Size::ZERO
                        && pair_offsets[1] == *offset
                        && align == pair.align
                        && size == pair.size
                    {
                        // We can use `ScalarPair` only when it matches our
                        // already computed layout (including `#[repr(C)]`).
                        abi = pair.abi;
                    }
                }
            }

            // If we pick a "clever" (by-value) ABI, we might have to adjust the ABI of the
            // variants to ensure they are consistent. This is because a downcast is
            // semantically a NOP, and thus should not affect layout.
            if matches!(abi, Abi::Scalar(..) | Abi::ScalarPair(..)) {
                for variant in &mut layout_variants {
                    // We only do this for variants with fields; the others are not accessed anyway.
                    // Also do not overwrite any already existing "clever" ABIs.
                    if variant.fields.count() > 0 && matches!(variant.abi, Abi::Aggregate { .. }) {
                        variant.abi = abi;
                        // Also need to bump up the size and alignment, so that the entire value fits in here.
                        variant.size = cmp::max(variant.size, size);
                        variant.align.abi = cmp::max(variant.align.abi, align.abi);
                    }
                }
            }

            let largest_niche = Niche::from_scalar(dl, Size::ZERO, tag);

            let tagged_layout = LayoutS {
                variants: Variants::Multiple {
                    tag,
                    tag_encoding: TagEncoding::Direct,
                    tag_field: 0,
                    variants: IndexVec::new(),
                },
                fields: FieldsShape::Arbitrary { offsets: vec![Size::ZERO], memory_index: vec![0] },
                largest_niche,
                abi,
                align,
                size,
            };

            let tagged_layout = TmpLayout { layout: tagged_layout, variants: layout_variants };

            let mut best_layout = match (tagged_layout, niche_filling_layout) {
                (tl, Some(nl)) => {
                    // Pick the smaller layout; otherwise,
                    // pick the layout with the larger niche; otherwise,
                    // pick tagged as it has simpler codegen.
                    use Ordering::*;
                    let niche_size = |tmp_l: &TmpLayout<'_>| {
                        tmp_l.layout.largest_niche.map_or(0, |n| n.available(dl))
                    };
                    match (
                        tl.layout.size.cmp(&nl.layout.size),
                        niche_size(&tl).cmp(&niche_size(&nl)),
                    ) {
                        (Greater, _) => nl,
                        (Equal, Less) => nl,
                        _ => tl,
                    }
                }
                (tl, None) => tl,
            };

            // Now we can intern the variant layouts and store them in the enum layout.
            best_layout.layout.variants = match best_layout.layout.variants {
                Variants::Multiple { tag, tag_encoding, tag_field, .. } => Variants::Multiple {
                    tag,
                    tag_encoding,
                    tag_field,
                    variants: best_layout
                        .variants
                        .into_iter()
                        .map(|layout| tcx.intern_layout(layout))
                        .collect(),
                },
                _ => bug!(),
            };

            tcx.intern_layout(best_layout.layout)
        }

        // Types with no meaningful known layout.
        ty::Projection(_) | ty::Opaque(..) => {
            // NOTE(eddyb) `layout_of` query should've normalized these away,
            // if that was possible, so there's no reason to try again here.
            return Err(LayoutError::Unknown(ty));
        }

        ty::Placeholder(..) | ty::GeneratorWitness(..) | ty::Infer(_) => {
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
fn generator_saved_local_eligibility<'tcx>(
    info: &GeneratorLayout<'tcx>,
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
    let discr_int_ty = discr_int.to_ty(tcx, false);
    let tag = Scalar::Initialized {
        value: Primitive::Int(discr_int, false),
        valid_range: WrappingRange { start: 0, end: max_discr },
    };
    let tag_layout = cx.tcx.intern_layout(LayoutS::scalar(cx, tag));
    let tag_layout = TyAndLayout { ty: discr_int_ty, layout: tag_layout };

    let promoted_layouts = ineligible_locals
        .iter()
        .map(|local| subst_field(info.field_tys[local]))
        .map(|ty| tcx.mk_maybe_uninit(ty))
        .map(|ty| cx.layout_of(ty));
    let prefix_layouts = substs
        .as_generator()
        .prefix_tys()
        .map(|ty| cx.layout_of(ty))
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
                .map(|local| subst_field(info.field_tys[*local]));

            let mut variant = univariant_uninterned(
                cx,
                ty,
                &variant_only_tys.map(|ty| cx.layout_of(ty)).collect::<Result<Vec<_>, _>>()?,
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
            Ok(tcx.intern_layout(variant))
        })
        .collect::<Result<IndexVec<VariantIdx, _>, _>>()?;

    size = size.align_to(align.abi);

    let abi = if prefix.abi.is_uninhabited() || variants.iter().all(|v| v.abi().is_uninhabited()) {
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

    let adt_def = match *layout.ty.kind() {
        ty::Adt(ref adt_def, _) => {
            debug!("print-type-size t: `{:?}` process adt", layout.ty);
            adt_def
        }

        ty::Closure(..) => {
            debug!("print-type-size t: `{:?}` record closure", layout.ty);
            record(DataTypeKind::Closure, false, None, vec![]);
            return;
        }

        _ => {
            debug!("print-type-size t: `{:?}` skip non-nominal", layout.ty);
            return;
        }
    };

    let adt_kind = adt_def.adt_kind();
    let adt_packed = adt_def.repr().pack.is_some();

    let build_variant_info = |n: Option<Symbol>, flds: &[Symbol], layout: TyAndLayout<'tcx>| {
        let mut min_size = Size::ZERO;
        let field_info: Vec<_> = flds
            .iter()
            .enumerate()
            .map(|(i, &name)| {
                let field_layout = layout.field(cx, i);
                let offset = layout.fields.offset(i);
                let field_end = offset + field_layout.size;
                if min_size < field_end {
                    min_size = field_end;
                }
                FieldInfo {
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
                record(
                    adt_kind.into(),
                    adt_packed,
                    None,
                    vec![build_variant_info(Some(variant_def.name), &fields, layout)],
                );
            } else {
                // (This case arises for *empty* enums; so give it
                // zero variants.)
                record(adt_kind.into(), adt_packed, None, vec![]);
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
            record(
                adt_kind.into(),
                adt_packed,
                match tag_encoding {
                    TagEncoding::Direct => Some(tag.size(cx)),
                    _ => None,
                },
                variant_infos,
            );
        }
    }
}
