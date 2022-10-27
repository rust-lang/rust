//! Compute the binary representation of structs, unions and enums

use std::{
    cmp::{self, Ordering},
    iter,
    num::NonZeroUsize,
    ops::Bound,
};

use chalk_ir::TyKind;
use hir_def::{
    adt::VariantData,
    layout::{
        Abi, AbiAndPrefAlign, Align, FieldsShape, Integer, Layout, LayoutError, Niche, Primitive,
        ReprOptions, Scalar, Size, StructKind, TagEncoding, TargetDataLayout, Variants,
        WrappingRange,
    },
    AdtId, EnumVariantId, LocalEnumVariantId, UnionId, VariantId,
};
use la_arena::{ArenaMap, RawIdx};

struct X(Option<NonZeroUsize>);

use crate::{
    db::HirDatabase,
    lang_items::is_unsafe_cell,
    layout::{field_ty, scalar_unit},
    Interner, Substitution,
};

use super::layout_of_ty;

pub fn layout_of_adt_query(
    db: &dyn HirDatabase,
    def: AdtId,
    subst: Substitution,
) -> Result<Layout, LayoutError> {
    let handle_variant = |def: VariantId, var: &VariantData| {
        var.fields()
            .iter()
            .map(|(fd, _)| layout_of_ty(db, &field_ty(db, def, fd, &subst)))
            .collect::<Result<Vec<_>, _>>()
    };
    fn struct_variant_idx() -> LocalEnumVariantId {
        LocalEnumVariantId::from_raw(RawIdx::from(0))
    }
    let (variants, is_enum, repr) = match def {
        AdtId::StructId(s) => {
            let data = db.struct_data(s);
            let mut r = ArenaMap::new();
            r.insert(struct_variant_idx(), handle_variant(s.into(), &data.variant_data)?);
            (r, false, data.repr.unwrap_or_default())
        }
        AdtId::UnionId(id) => return layout_of_union(db, id, &subst),
        AdtId::EnumId(e) => {
            let data = db.enum_data(e);
            let r = data
                .variants
                .iter()
                .map(|(idx, v)| {
                    Ok((
                        idx,
                        handle_variant(
                            EnumVariantId { parent: e, local_id: idx }.into(),
                            &v.variant_data,
                        )?,
                    ))
                })
                .collect::<Result<_, _>>()?;
            (r, true, data.repr.unwrap_or_default())
        }
    };

    // A variant is absent if it's uninhabited and only has ZST fields.
    // Present uninhabited variants only require space for their fields,
    // but *not* an encoding of the discriminant (e.g., a tag value).
    // See issue #49298 for more details on the need to leave space
    // for non-ZST uninhabited data (mostly partial initialization).
    let absent = |fields: &[Layout]| {
        let uninhabited = fields.iter().any(|f| f.abi.is_uninhabited());
        let is_zst = fields.iter().all(|f| f.is_zst());
        uninhabited && is_zst
    };
    let (present_first, present_second) = {
        let mut present_variants =
            variants.iter().filter_map(|(i, v)| if absent(v) { None } else { Some(i) });
        (present_variants.next(), present_variants.next())
    };
    let present_first = match present_first {
        Some(present_first) => present_first,
        // Uninhabited because it has no variants, or only absent ones.
        None if is_enum => return layout_of_ty(db, &TyKind::Never.intern(Interner)),
        // If it's a struct, still compute a layout so that we can still compute the
        // field offsets.
        None => struct_variant_idx(),
    };

    let is_univariant = !is_enum ||
                    // Only one variant is present.
                    (present_second.is_none() &&
                        // Representation optimizations are allowed.
                        !repr.inhibit_enum_layout_opt());
    let dl = &*db.current_target_data_layout();

    if is_univariant {
        // Struct, or univariant enum equivalent to a struct.
        // (Typechecking will reject discriminant-sizing attrs.)

        let v = present_first;
        let kind = if is_enum || variants[v].is_empty() {
            StructKind::AlwaysSized
        } else {
            let always_sized = !variants[v].last().unwrap().is_unsized();
            if !always_sized {
                StructKind::MaybeUnsized
            } else {
                StructKind::AlwaysSized
            }
        };

        let mut st = univariant(dl, &variants[v], &repr, kind)?;
        st.variants = Variants::Single;

        if is_unsafe_cell(def, db) {
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
            return Ok(st);
        }

        let (start, end) = layout_scalar_valid_range(db, def);
        match st.abi {
            Abi::Scalar(ref mut scalar) | Abi::ScalarPair(ref mut scalar, _) => {
                if let Bound::Included(start) = start {
                    let valid_range = scalar.valid_range_mut();
                    valid_range.start = start;
                }
                if let Bound::Included(end) = end {
                    let valid_range = scalar.valid_range_mut();
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
            _ => user_error!("nonscalar layout for layout_scalar_valid_range"),
        }

        return Ok(st);
    }

    // Until we've decided whether to use the tagged or
    // niche filling LayoutS, we don't want to intern the
    // variant layouts, so we can't store them in the
    // overall LayoutS. Store the overall LayoutS
    // and the variant LayoutSs here until then.
    struct TmpLayout {
        layout: Layout,
        variants: ArenaMap<LocalEnumVariantId, Layout>,
    }

    let calculate_niche_filling_layout = || -> Result<Option<TmpLayout>, LayoutError> {
        // The current code for niche-filling relies on variant indices
        // instead of actual discriminants, so enums with
        // explicit discriminants (RFC #2363) would misbehave.
        if repr.inhibit_enum_layout_opt()
        // FIXME: bring these codes back
        // || def
        //     .variants()
        //     .iter_enumerated()
        //     .any(|(i, v)| v.discr != ty::VariantDiscr::Relative(i.as_u32()))
        {
            return Ok(None);
        }

        if variants.iter().count() < 2 {
            return Ok(None);
        }

        let mut align = dl.aggregate_align;
        let mut variant_layouts = variants
            .iter()
            .map(|(j, v)| {
                let mut st = univariant(dl, v, &repr, StructKind::AlwaysSized)?;
                st.variants = Variants::Single;

                align = align.max(st.align);

                Ok((j, st))
            })
            .collect::<Result<ArenaMap<_, _>, _>>()?;

        let largest_variant_index = match variant_layouts
            .iter()
            .max_by_key(|(_i, layout)| layout.size.bytes())
            .map(|(i, _layout)| i)
        {
            None => return Ok(None),
            Some(i) => i,
        };

        let count = variants
            .iter()
            .map(|(i, _)| i)
            .filter(|x| *x != largest_variant_index && !absent(&variants[*x]))
            .count() as u128;

        // Find the field with the largest niche
        let (field_index, niche, (niche_start, niche_scalar)) = match variants
            [largest_variant_index]
            .iter()
            .enumerate()
            .filter_map(|(j, field)| Some((j, field.largest_niche?)))
            .max_by_key(|(_, niche)| niche.available(dl))
            .and_then(|(j, niche)| Some((j, niche, niche.reserve(dl, count)?)))
        {
            None => return Ok(None),
            Some(x) => x,
        };

        let niche_offset =
            niche.offset + variant_layouts[largest_variant_index].fields.offset(field_index, dl);
        let niche_size = niche.value.size(dl);
        let size = variant_layouts[largest_variant_index].size.align_to(align.abi);

        let all_variants_fit = variant_layouts.iter_mut().all(|(i, layout)| {
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
            .iter()
            .all(|(i, layout)| i == largest_variant_index || layout.size == Size::ZERO);
        let same_size = size == variant_layouts[largest_variant_index].size;
        let same_align = align == variant_layouts[largest_variant_index].align;

        let abi = if variant_layouts.iter().all(|(_, v)| v.abi.is_uninhabited()) {
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

        let layout = Layout {
            variants: Variants::Multiple {
                tag: niche_scalar,
                tag_encoding: TagEncoding::Niche {
                    untagged_variant: largest_variant_index,
                    niche_start,
                },
                tag_field: 0,
                variants: ArenaMap::new(),
            },
            fields: FieldsShape::Arbitrary { offsets: vec![niche_offset], memory_index: vec![0] },
            abi,
            largest_niche,
            size,
            align,
        };

        Ok(Some(TmpLayout { layout, variants: variant_layouts }))
    };

    let niche_filling_layout = calculate_niche_filling_layout()?;

    let (mut min, mut max) = (i128::MAX, i128::MIN);
    // FIXME: bring these back
    // let discr_type = repr.discr_type();
    // let bits = Integer::from_attr(dl, discr_type).size().bits();
    // for (i, discr) in def.discriminants(tcx) {
    //     if variants[i].iter().any(|f| f.abi.is_uninhabited()) {
    //         continue;
    //     }
    //     let mut x = discr.val as i128;
    //     if discr_type.is_signed() {
    //         // sign extend the raw representation to be an i128
    //         x = (x << (128 - bits)) >> (128 - bits);
    //     }
    //     if x < min {
    //         min = x;
    //     }
    //     if x > max {
    //         max = x;
    //     }
    // }
    // We might have no inhabited variants, so pretend there's at least one.
    if (min, max) == (i128::MAX, i128::MIN) {
        min = 0;
        max = 0;
    }
    assert!(min <= max, "discriminant range is {}...{}", min, max);
    let (min_ity, signed) = Integer::repr_discr(dl, &repr, min, max)?;

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
    if repr.c() {
        for (_, fields) in variants.iter() {
            for field in fields {
                prefix_align = prefix_align.max(field.align.abi);
            }
        }
    }

    // Create the set of structs that represent each variant.
    let mut layout_variants = variants
        .iter()
        .map(|(i, field_layouts)| {
            let mut st = univariant(
                dl,
                &field_layouts,
                &repr,
                StructKind::Prefixed(min_ity.size(), prefix_align),
            )?;
            st.variants = Variants::Single;
            // Find the first field we can't move later
            // to make room for a larger discriminant.
            for field in st.fields.index_by_increasing_offset().map(|j| &field_layouts[j]) {
                if !field.is_zst() || field.align.abi.bytes() != 1 {
                    start_align = start_align.min(field.align.abi);
                    break;
                }
            }
            size = cmp::max(size, st.size);
            align = align.max(st.align);
            Ok((i, st))
        })
        .collect::<Result<ArenaMap<_, _>, _>>()?;

    // Align the maximum variant size to the largest alignment.
    size = size.align_to(align.abi);

    if size.bytes() >= dl.obj_size_bound() {
        return Err(LayoutError::SizeOverflow);
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
        // Patch up the variants' first few fields.
        let old_ity_size = min_ity.size();
        let new_ity_size = ity.size();
        for (_, variant) in layout_variants.iter_mut() {
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
                _ => user_error!("bug"),
            }
        }
    }

    let tag_mask = ity.size().unsigned_int_max();
    let tag = Scalar::Initialized {
        value: Primitive::Int(ity, signed),
        valid_range: WrappingRange {
            start: (min as u128 & tag_mask),
            end: (max as u128 & tag_mask),
        },
    };
    let mut abi = Abi::Aggregate { sized: true };

    if layout_variants.iter().all(|(_, v)| v.abi.is_uninhabited()) {
        abi = Abi::Uninhabited;
    } else if tag.size(dl) == size {
        // Make sure we only use scalar layout when the enum is entirely its
        // own tag (i.e. it has no padding nor any non-ZST variant fields).
        abi = Abi::Scalar(tag);
    } else {
        // Try to use a ScalarPair for all tagged enums.
        let mut common_prim = None;
        let mut common_prim_initialized_in_all_variants = true;
        for ((_, field_layouts), (_, layout_variant)) in
            iter::zip(variants.iter(), layout_variants.iter())
        {
            let offsets = match layout_variant.fields {
                FieldsShape::Arbitrary { ref offsets, .. } => offsets,
                _ => user_error!("bug"),
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
                scalar_unit(dl, prim)
            } else {
                // Common prim might be uninit.
                Scalar::Union { value: prim }
            };
            let pair = scalar_pair(dl, tag, prim_scalar);
            let pair_offsets = match pair.fields {
                FieldsShape::Arbitrary { ref offsets, ref memory_index } => {
                    assert_eq!(memory_index, &[0, 1]);
                    offsets
                }
                _ => user_error!("bug"),
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
        for (_, variant) in layout_variants.iter_mut() {
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

    let tagged_layout = Layout {
        variants: Variants::Multiple {
            tag,
            tag_encoding: TagEncoding::Direct,
            tag_field: 0,
            variants: ArenaMap::new(),
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
            let niche_size =
                |tmp_l: &TmpLayout| tmp_l.layout.largest_niche.map_or(0, |n| n.available(dl));
            match (tl.layout.size.cmp(&nl.layout.size), niche_size(&tl).cmp(&niche_size(&nl))) {
                (Greater, _) => nl,
                (Equal, Less) => nl,
                _ => tl,
            }
        }
        (tl, None) => tl,
    };

    // Now we can intern the variant layouts and store them in the enum layout.
    best_layout.layout.variants = match best_layout.layout.variants {
        Variants::Multiple { tag, tag_encoding, tag_field, .. } => {
            Variants::Multiple { tag, tag_encoding, tag_field, variants: best_layout.variants }
        }
        _ => user_error!("bug"),
    };

    Ok(best_layout.layout)
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

pub(crate) fn univariant(
    dl: &TargetDataLayout,
    fields: &[Layout],
    repr: &ReprOptions,
    kind: StructKind,
) -> Result<Layout, LayoutError> {
    let pack = repr.pack;
    if pack.is_some() && repr.align.is_some() {
        user_error!("Struct can not be packed and aligned");
    }

    let mut align = if pack.is_some() { dl.i8_align } else { dl.aggregate_align };

    let mut inverse_memory_index: Vec<u32> = (0..fields.len() as u32).collect();

    let optimize = !repr.inhibit_struct_field_reordering_opt();
    if optimize {
        let end = if let StructKind::MaybeUnsized = kind { fields.len() - 1 } else { fields.len() };
        let optimizing = &mut inverse_memory_index[..end];
        let field_align = |f: &Layout| {
            if let Some(pack) = pack {
                f.align.abi.min(pack)
            } else {
                f.align.abi
            }
        };

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
        let field = &fields[i as usize];
        if !sized {
            user_error!("Unsized field is not last field");
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

        offsets[i as usize] = offset;

        if let Some(mut niche) = field.largest_niche {
            let available = niche.available(dl);
            if available > largest_niche_available {
                largest_niche_available = available;
                niche.offset =
                    niche.offset.checked_add(offset, dl).ok_or(LayoutError::SizeOverflow)?;
                largest_niche = Some(niche);
            }
        }

        offset = offset.checked_add(field.size, dl).ok_or(LayoutError::SizeOverflow)?;
    }

    if let Some(repr_align) = repr.align {
        align = align.max(AbiAndPrefAlign::new(repr_align));
    }

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
                        let pair = scalar_pair(dl, a, b);
                        let pair_offsets = match pair.fields {
                            FieldsShape::Arbitrary { ref offsets, .. } => offsets,
                            _ => unreachable!(),
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

    Ok(Layout {
        variants: Variants::Single,
        fields: FieldsShape::Arbitrary { offsets, memory_index },
        abi,
        largest_niche,
        align,
        size,
    })
}

fn layout_of_union(
    db: &dyn HirDatabase,
    id: UnionId,
    subst: &Substitution,
) -> Result<Layout, LayoutError> {
    let dl = &*db.current_target_data_layout();

    let union_data = db.union_data(id);

    let repr = union_data.repr.unwrap_or_default();
    let fields = union_data.variant_data.fields();

    if repr.pack.is_some() && repr.align.is_some() {
        user_error!("union cannot be packed and aligned");
    }

    let mut align = if repr.pack.is_some() { dl.i8_align } else { dl.aggregate_align };
    if let Some(repr_align) = repr.align {
        align = align.max(AbiAndPrefAlign::new(repr_align));
    }

    let optimize = !repr.inhibit_union_abi_opt();
    let mut size = Size::ZERO;
    let mut abi = Abi::Aggregate { sized: true };
    for (fd, _) in fields.iter() {
        let field_ty = field_ty(db, id.into(), fd, subst);
        let field = layout_of_ty(db, &field_ty)?;
        if field.is_unsized() {
            user_error!("unsized union field");
        }
        // If all non-ZST fields have the same ABI, forward this ABI
        if optimize && !field.is_zst() {
            // Discard valid range information and allow undef
            let field_abi = match field.abi {
                Abi::Scalar(x) => Abi::Scalar(x.to_union()),
                Abi::ScalarPair(x, y) => Abi::ScalarPair(x.to_union(), y.to_union()),
                Abi::Vector { element: x, count } => Abi::Vector { element: x.to_union(), count },
                Abi::Uninhabited | Abi::Aggregate { .. } => Abi::Aggregate { sized: true },
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

    if let Some(pack) = repr.pack {
        align = align.min(AbiAndPrefAlign::new(pack));
    }

    Ok(Layout {
        variants: Variants::Single,
        fields: FieldsShape::Union(
            NonZeroUsize::new(fields.len())
                .ok_or(LayoutError::UserError("union with zero fields".to_string()))?,
        ),
        abi,
        largest_niche: None,
        align,
        size: size.align_to(align.abi),
    })
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

fn scalar_pair(dl: &TargetDataLayout, a: Scalar, b: Scalar) -> Layout {
    let b_align = b.align(dl);
    let align = a.align(dl).max(b_align).max(dl.aggregate_align);
    let b_offset = a.size(dl).align_to(b_align.abi);
    let size = b_offset.checked_add(b.size(dl), dl).unwrap().align_to(align.abi);

    // HACK(nox): We iter on `b` and then `a` because `max_by_key`
    // returns the last maximum.
    let largest_niche = Niche::from_scalar(dl, b_offset, b)
        .into_iter()
        .chain(Niche::from_scalar(dl, Size::ZERO, a))
        .max_by_key(|niche| niche.available(dl));

    Layout {
        variants: Variants::Single,
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
