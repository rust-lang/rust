use std::ops::Deref;
use std::range::{RangeFrom, RangeInclusive, RangeToInclusive};
use std::{cmp, fmt, iter};

use rustc_index::Idx;
use tracing::trace;

use super::{LayoutCalculator, LayoutCalculatorError, LayoutCalculatorResult, absent};
use crate::{
    AbiAlign, Align, BackendRepr, FieldsShape, HasDataLayout, IndexSlice, IndexVec, Integer,
    LayoutData, Niche, Primitive, ReprOptions, Scalar, Size, StructKind, TagEncoding,
    VariantLayout, Variants, WrappingRange,
};

pub(super) fn layout_of_enum<'a, Cx: HasDataLayout, FieldIdx, VariantIdx, F>(
    calculator: &LayoutCalculator<Cx>,
    repr: &ReprOptions,
    variants: &IndexSlice<VariantIdx, IndexVec<FieldIdx, F>>,
    discr_range_of_repr: impl Fn(RangeFrom<i128>, RangeToInclusive<u128>) -> (Integer, bool),
    discriminants: impl Iterator<Item = (VariantIdx, u128)>,
) -> LayoutCalculatorResult<FieldIdx, VariantIdx, F>
where
    FieldIdx: Idx,
    VariantIdx: Idx,
    F: Deref<Target = &'a LayoutData<FieldIdx, VariantIdx>> + fmt::Debug + Copy,
{
    let dl = calculator.cx.data_layout();
    // bail if the enum has an incoherent repr that cannot be computed
    if repr.packed() {
        return Err(LayoutCalculatorError::ReprConflict);
    }

    let niche_filling_layout = calculate_niche_filling_layout(calculator, repr, variants);

    let tagged_layout =
        calculate_tagged_layout(calculator, repr, variants, discr_range_of_repr, discriminants)?;

    let best_layout = match (tagged_layout, niche_filling_layout) {
        (tl, Some(nl)) => {
            // Pick the smaller layout; otherwise,
            // pick the layout with the larger niche; otherwise,
            // pick tagged as it has simpler codegen.
            use cmp::Ordering::*;
            let niche_size = |l: &LayoutData<FieldIdx, VariantIdx>| {
                l.largest_niche.map_or(0, |n| n.available(dl))
            };
            match (tl.size.cmp(&nl.size), niche_size(&tl).cmp(&niche_size(&nl))) {
                (Greater, _) => nl,
                (Equal, Less) => nl,
                _ => tl,
            }
        }
        (tl, None) => tl,
    };

    Ok(best_layout)
}

fn calculate_tagged_layout<'a, Cx: HasDataLayout, FieldIdx, VariantIdx, F>(
    calculator: &LayoutCalculator<Cx>,
    repr: &ReprOptions,
    variants: &IndexSlice<VariantIdx, IndexVec<FieldIdx, F>>,
    discr_range_of_repr: impl Fn(RangeFrom<i128>, RangeToInclusive<u128>) -> (Integer, bool),
    discriminants: impl Iterator<Item = (VariantIdx, u128)>,
) -> LayoutCalculatorResult<FieldIdx, VariantIdx, F>
where
    FieldIdx: Idx,
    VariantIdx: Idx,
    F: Deref<Target = &'a LayoutData<FieldIdx, VariantIdx>> + fmt::Debug + Copy,
{
    let dl = calculator.cx.data_layout();
    let discr_type = repr.discr_type();
    let discr_size = Integer::from_attr(dl, discr_type).size();

    let necessary_discriminants: Vec<u128> = discriminants
        .filter(|&(i, _)| repr.c() || variants[i].iter().all(|f| !f.is_uninhabited()))
        .map(|(_, val)| val)
        .collect();

    // When picking the integer to use, we respect how the discriminants were written
    // in the original rust code, rather than looking only at the bit pattern.
    let (min_negative, max_positive): (i128, u128) = if discr_type.is_signed() {
        necessary_discriminants.iter().copied().map(|val| discr_size.sign_extend(val)).fold(
            (0_i128, 0_u128),
            |(min, max), val| {
                if let Ok(val) = u128::try_from(val) {
                    (min, max.max(val))
                } else {
                    (min.min(val), max)
                }
            },
        )
    } else {
        // We might have no inhabited variants, so pretend there's at least one.
        (0, necessary_discriminants.iter().copied().max().unwrap_or(0))
    };
    trace!(?min_negative, ?max_positive);

    let (min_ity, signed) = discr_range_of_repr(
        RangeFrom { start: min_negative },
        RangeToInclusive { last: max_positive },
    ); //Integer::discr_range_of_repr(tcx, ty, &repr, min, max);

    let mut align = dl.aggregate_align;
    let mut max_repr_align = repr.align;
    let mut unadjusted_abi_align = align;
    let mut combined_seed = repr.field_shuffle_seed;

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
    if repr.c() {
        for fields in variants {
            for field in fields {
                prefix_align = prefix_align.max(field.align.abi);
            }
        }
    }

    // Create the set of structs that represent each variant.
    let mut layout_variants = variants
        .iter()
        .map(|field_layouts| {
            let st = calculator.layout_of_univariant(
                field_layouts,
                repr,
                StructKind::Prefixed(min_ity.size(), prefix_align),
            )?;
            // Find the first field we can't move later
            // to make room for a larger discriminant.
            for field_idx in st.fields.index_by_increasing_offset() {
                let field = &field_layouts[FieldIdx::new(field_idx)];
                if !field.is_1zst() {
                    start_align = start_align.min(field.align.abi);
                    break;
                }
            }
            size = cmp::max(size, st.size);
            align = align.max(st.align.abi);
            max_repr_align = max_repr_align.max(st.max_repr_align);
            unadjusted_abi_align = unadjusted_abi_align.max(st.unadjusted_abi_align);
            combined_seed = combined_seed.wrapping_add(st.randomization_seed);
            Ok(VariantLayout::from_layout(st))
        })
        .collect::<Result<IndexVec<VariantIdx, _>, _>>()?;

    // Align the maximum variant size to the largest alignment.
    size = size.align_to(align);

    // FIXME(oli-obk): deduplicate and harden these checks
    if size.bytes() >= dl.obj_size_bound() {
        return Err(LayoutCalculatorError::SizeOverflow);
    }

    let typeck_ity = Integer::from_attr(dl, repr.discr_type());
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
        panic!(
            "layout decided on a larger discriminant type ({min_ity:?}) than typeck ({typeck_ity:?})"
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
    let mut ity = if repr.c() || repr.int.is_some() {
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
            for i in &mut variant.field_offsets {
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
    }

    let tag_valid_range = {
        let tag_size = ity.size();
        let tags = necessary_discriminants.into_iter().map(|d| tag_size.truncate(d));
        WrappingRange::smallest_range_containing(tags, tag_size)
            // We might have no inhabited variants, so pretend there's at least one.
            .unwrap_or(WrappingRange { start: 0, end: 0 })
    };
    let tag =
        Scalar::Initialized { value: Primitive::Int(ity, signed), valid_range: tag_valid_range };
    let mut abi = BackendRepr::Memory { sized: true };

    let uninhabited = layout_variants.iter().all(|v| v.is_uninhabited());
    if tag.size(dl) == size {
        // Make sure we only use scalar layout when the enum is entirely its
        // own tag (i.e. it has no padding nor any non-ZST variant fields).
        abi = BackendRepr::Scalar(tag);
    } else {
        // Try to use a ScalarPair for all tagged enums.
        // That's possible only if we can find a common primitive type for all variants.
        let mut common_prim = None;
        let mut common_prim_initialized_in_all_variants = true;
        for (field_layouts, layout_variant) in iter::zip(variants, &layout_variants) {
            // We skip *all* ZST here and later check if we are good in terms of alignment.
            // This lets us handle some cases involving aligned ZST.
            let mut fields =
                iter::zip(field_layouts, &layout_variant.field_offsets).filter(|p| !p.0.is_zst());
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
            let prim = match field.backend_repr {
                BackendRepr::Scalar(scalar) => {
                    common_prim_initialized_in_all_variants &=
                        matches!(scalar, Scalar::Initialized { .. });
                    scalar.primitive()
                }
                _ => {
                    common_prim = None;
                    break;
                }
            };
            if let Some((old_prim, common_offset)) = common_prim {
                // All variants must be at the same offset
                if offset != common_offset {
                    common_prim = None;
                    break;
                }
                // This is pretty conservative. We could go fancier
                // by realising that (u8, u8) could just cohabit with
                // u16 or even u32.
                let new_prim = match (old_prim, prim) {
                    // Allow all identical primitives.
                    (x, y) if x == y => x,
                    // Allow integers of the same size with differing signedness.
                    // We arbitrarily choose the signedness of the first variant.
                    (p @ Primitive::Int(x, _), Primitive::Int(y, _)) if x == y => p,
                    // Allow integers mixed with pointers of the same layout.
                    // We must represent this using a pointer, to avoid
                    // roundtripping pointers through ptrtoint/inttoptr.
                    (p @ Primitive::Pointer(_), i @ Primitive::Int(..))
                    | (i @ Primitive::Int(..), p @ Primitive::Pointer(_))
                        if p.size(dl) == i.size(dl)
                            && p.default_align(dl) == i.default_align(dl) =>
                    {
                        p
                    }
                    _ => {
                        common_prim = None;
                        break;
                    }
                };
                // We may be updating the primitive here, for example from int->ptr.
                common_prim = Some((new_prim, common_offset));
            } else {
                common_prim = Some((prim, offset));
            }
        }
        if let Some((prim, offset)) = common_prim {
            let prim_scalar = if common_prim_initialized_in_all_variants {
                let size = prim.size(dl);
                assert!(size.bits() <= 128);
                Scalar::Initialized { value: prim, valid_range: WrappingRange::full(size) }
            } else {
                // Common prim might be uninit.
                Scalar::Union { value: prim }
            };
            let pair =
                LayoutData::<FieldIdx, VariantIdx>::scalar_pair(&calculator.cx, tag, prim_scalar);
            let pair_offsets = match pair.fields {
                FieldsShape::Arbitrary { ref offsets, ref in_memory_order } => {
                    assert_eq!(in_memory_order.raw, [FieldIdx::new(0), FieldIdx::new(1)]);
                    offsets
                }
                _ => panic!("encountered a non-arbitrary layout during enum layout"),
            };
            if pair_offsets[FieldIdx::new(0)] == Size::ZERO
                && pair_offsets[FieldIdx::new(1)] == *offset
                && align == pair.align.abi
                && size == pair.size
            {
                // We can use `ScalarPair` only when it matches our
                // already computed layout (including `#[repr(C)]`).
                abi = pair.backend_repr;
            }
        }
    }

    // If we pick a "clever" (by-value) ABI, we might have to adjust the ABI of the
    // variants to ensure they are consistent. This is because a downcast is
    // semantically a NOP, and thus should not affect layout.
    if matches!(abi, BackendRepr::Scalar(..) | BackendRepr::ScalarPair { .. }) {
        for variant in &mut layout_variants {
            // We only do this for variants with fields; the others are not accessed anyway.
            // Also do not overwrite any already existing "clever" ABIs.
            if matches!(variant.backend_repr, BackendRepr::Memory { .. } if variant.has_fields()) {
                variant.backend_repr = abi;
                // Also need to bump up the size, so that the entire value fits in here.
                variant.size = cmp::max(variant.size, size);
            }
        }
    }

    let largest_niche = Niche::from_scalar(dl, Size::ZERO, tag);

    Ok(LayoutData {
        variants: Variants::Multiple {
            tag,
            tag_encoding: TagEncoding::Direct,
            tag_field: FieldIdx::new(0),
            variants: layout_variants,
        },
        fields: FieldsShape::Arbitrary {
            offsets: [Size::ZERO].into(),
            in_memory_order: [FieldIdx::new(0)].into(),
        },
        largest_niche,
        uninhabited,
        backend_repr: abi,
        align: AbiAlign::new(align),
        size,
        max_repr_align,
        unadjusted_abi_align,
        repr_c: repr.c(),
        randomization_seed: combined_seed,
    })
}

struct VariantLayoutInfo {
    align_abi: Align,
}

fn try_fixup_non_niche_variants<VariantIdx: Idx, FieldIdx: Idx>(
    mut variant_layouts: IndexVec<VariantIdx, VariantLayout<FieldIdx>>,
    variants_info: &IndexSlice<VariantIdx, VariantLayoutInfo>,
    largest_variant_index: VariantIdx,
    niche_offset: Size,
    niche_size: Size,
    size: Size,
) -> Option<IndexVec<VariantIdx, VariantLayout<FieldIdx>>> {
    for (i, layout) in variant_layouts.iter_enumerated_mut() {
        if i == largest_variant_index {
            continue;
        }

        layout.largest_niche = None;

        if layout.size <= niche_offset {
            // This variant will fit before the niche.
            continue;
        }

        // Determine if it'll fit after the niche.
        let this_align = variants_info[i].align_abi;
        let this_offset = (niche_offset + niche_size).align_to(this_align);

        if this_offset + layout.size > size {
            return None;
        }

        // It'll fit, but we need to make some adjustments.
        for offset in layout.field_offsets.iter_mut() {
            *offset += this_offset;
        }

        // It can't be a Scalar or ScalarPair because the offset isn't 0.
        if !layout.is_uninhabited() {
            layout.backend_repr = BackendRepr::Memory { sized: true };
        }
        layout.size += this_offset;
    }
    Some(variant_layouts)
}

fn calculate_niche_abi<FieldIdx: Idx, VariantIdx: Idx>(
    variant_layouts: &IndexSlice<VariantIdx, VariantLayout<FieldIdx>>,
    variants_info: &IndexSlice<VariantIdx, VariantLayoutInfo>,
    niche_scalar: Scalar,
    largest_variant_index: VariantIdx,
    size: Size,
    align: Align,
    niche_offset: Size,
) -> BackendRepr {
    let others_zst = variant_layouts
        .iter_enumerated()
        .all(|(i, layout)| i == largest_variant_index || layout.size == Size::ZERO);
    let same_size = size == variant_layouts[largest_variant_index].size;
    let same_align = align == variants_info[largest_variant_index].align_abi;

    if same_size && same_align && others_zst {
        match variant_layouts[largest_variant_index].backend_repr {
            // When the total alignment and size match, we can use the
            // same ABI as the scalar variant with the reserved niche.
            BackendRepr::Scalar(_) => BackendRepr::Scalar(niche_scalar),
            BackendRepr::ScalarPair { a: first, b: second, b_offset } => {
                // Only the niche is guaranteed to be initialised,
                // so use union layouts for the other primitive.
                //
                // How can this be nonzero when everything else is a ZST? `others_zst` is true here
                if niche_offset == Size::ZERO {
                    BackendRepr::ScalarPair { a: niche_scalar, b: second.to_union(), b_offset }
                } else {
                    BackendRepr::ScalarPair { a: first.to_union(), b: niche_scalar, b_offset }
                }
            }
            _ => BackendRepr::Memory { sized: true },
        }
    } else {
        BackendRepr::Memory { sized: true }
    }
}

fn calculate_niche_filling_layout<'a, Cx: HasDataLayout, FieldIdx, VariantIdx, F>(
    calculator: &LayoutCalculator<Cx>,
    repr: &ReprOptions,
    variants: &IndexSlice<VariantIdx, IndexVec<FieldIdx, F>>,
) -> Option<LayoutData<FieldIdx, VariantIdx>>
where
    FieldIdx: Idx,
    VariantIdx: Idx,
    F: Deref<Target = &'a LayoutData<FieldIdx, VariantIdx>> + fmt::Debug + Copy,
{
    let dl = calculator.cx.data_layout();
    if repr.inhibit_enum_layout_opt() {
        return None;
    }

    if variants.len() < 2 {
        return None;
    }

    let mut align = dl.aggregate_align;
    let mut max_repr_align = repr.align;
    let mut unadjusted_abi_align = align;
    let mut combined_seed = repr.field_shuffle_seed;

    let mut variants_info = IndexVec::<VariantIdx, _>::with_capacity(variants.len());
    let variant_layouts = variants
        .iter()
        .map(|v| {
            let st = calculator.layout_of_univariant(v, repr, StructKind::AlwaysSized).ok()?;

            variants_info.push(VariantLayoutInfo { align_abi: st.align.abi });

            align = align.max(st.align.abi);
            max_repr_align = max_repr_align.max(st.max_repr_align);
            unadjusted_abi_align = unadjusted_abi_align.max(st.unadjusted_abi_align);
            combined_seed = combined_seed.wrapping_add(st.randomization_seed);

            Some(VariantLayout::from_layout(st))
        })
        .collect::<Option<IndexVec<VariantIdx, _>>>()?;

    let largest_variant_index = variant_layouts
        .iter_enumerated()
        .max_by_key(|(_i, layout)| layout.size.bytes())
        .map(|(i, _layout)| i)?;

    let all_indices = variants.indices();
    let needs_disc =
        |index: VariantIdx| index != largest_variant_index && !absent(&variants[index]);
    let niche_variants = RangeInclusive {
        start: all_indices.clone().find(|v| needs_disc(*v)).unwrap(),
        last: all_indices.rev().find(|v| needs_disc(*v)).unwrap(),
    };

    let count = (niche_variants.last.index() as u128 - niche_variants.start.index() as u128) + 1;

    // Use the largest niche in the largest variant.
    let niche = variant_layouts[largest_variant_index].largest_niche?;
    let (niche_start, niche_scalar) = niche.reserve(dl, count)?;
    let size = variant_layouts[largest_variant_index].size.align_to(align);

    let variant_layouts = try_fixup_non_niche_variants(
        variant_layouts,
        &variants_info,
        largest_variant_index,
        niche.offset,
        niche.value.size(dl),
        size,
    )?;

    let abi = calculate_niche_abi(
        &variant_layouts,
        &variants_info,
        niche_scalar,
        largest_variant_index,
        size,
        align,
        niche.offset,
    );

    let layout = LayoutData {
        uninhabited: variant_layouts.iter().all(|v| v.is_uninhabited()),
        variants: Variants::Multiple {
            tag: niche_scalar,
            tag_encoding: TagEncoding::Niche {
                untagged_variant: largest_variant_index,
                niche_variants,
                niche_start,
            },
            tag_field: FieldIdx::new(0),
            variants: variant_layouts,
        },
        fields: FieldsShape::Arbitrary {
            offsets: [niche.offset].into(),
            in_memory_order: [FieldIdx::new(0)].into(),
        },
        backend_repr: abi,
        largest_niche: Niche::from_scalar(dl, niche.offset, niche_scalar),
        size,
        align: AbiAlign::new(align),
        max_repr_align,
        unadjusted_abi_align,
        repr_c: repr.c(),
        randomization_seed: combined_seed,
    };

    Some(layout)
}
