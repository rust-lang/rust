//! Coroutine layout logic.
//!
//! When laying out coroutines, we divide our saved local fields into two
//! categories: overlap-eligible and overlap-ineligible.
//!
//! Those fields which are ineligible for overlap go in a "prefix" at the
//! beginning of the layout, and always have space reserved for them.
//!
//! Overlap-eligible fields are only assigned to one variant, so we lay
//! those fields out for each variant and put them right after the
//! prefix.
//!
//! Finally, in the layout details, we point to the fields from the
//! variants they are assigned to. It is possible for some fields to be
//! included in multiple variants. No field ever "moves around" in the
//! layout; its offset is always the same.
//!
//! Also included in the layout are the upvars and the discriminant.
//! These are included as fields on the "outer" layout; they are not part
//! of any variant.

use std::iter;

use rustc_index::bit_set::{BitMatrix, DenseBitSet};
use rustc_index::{Idx, IndexSlice, IndexVec};
use tracing::{debug, trace};

use crate::{
    BackendRepr, FieldsShape, HasDataLayout, Integer, LayoutData, Primitive, ReprOptions, Scalar,
    StructKind, TagEncoding, Variants, WrappingRange,
};

/// Overlap eligibility and variant assignment for each CoroutineSavedLocal.
#[derive(Clone, Debug, PartialEq)]
enum SavedLocalEligibility<VariantIdx, FieldIdx> {
    Unassigned,
    Assigned(VariantIdx),
    Ineligible(Option<FieldIdx>),
}

/// Compute the eligibility and assignment of each local.
fn coroutine_saved_local_eligibility<VariantIdx: Idx, FieldIdx: Idx, LocalIdx: Idx>(
    nb_locals: usize,
    variant_fields: &IndexSlice<VariantIdx, IndexVec<FieldIdx, LocalIdx>>,
    storage_conflicts: &BitMatrix<LocalIdx, LocalIdx>,
) -> (DenseBitSet<LocalIdx>, IndexVec<LocalIdx, SavedLocalEligibility<VariantIdx, FieldIdx>>) {
    use SavedLocalEligibility::*;

    let mut assignments: IndexVec<LocalIdx, _> = IndexVec::from_elem_n(Unassigned, nb_locals);

    // The saved locals not eligible for overlap. These will get
    // "promoted" to the prefix of our coroutine.
    let mut ineligible_locals = DenseBitSet::new_empty(nb_locals);

    // Figure out which of our saved locals are fields in only
    // one variant. The rest are deemed ineligible for overlap.
    for (variant_index, fields) in variant_fields.iter_enumerated() {
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
                        local, variant_index, idx
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
    for local_a in storage_conflicts.rows() {
        let conflicts_a = storage_conflicts.count(local_a);
        if ineligible_locals.contains(local_a) {
            continue;
        }

        for local_b in storage_conflicts.iter(local_a) {
            // local_a and local_b are storage live at the same time, therefore they
            // cannot overlap in the coroutine layout. The only way to guarantee
            // this is if they are in the same variant, or one is ineligible
            // (which means it is stored in every variant).
            if ineligible_locals.contains(local_b) || assignments[local_a] == assignments[local_b] {
                continue;
            }

            // If they conflict, we will choose one to make ineligible.
            // This is not always optimal; it's just a greedy heuristic that
            // seems to produce good results most of the time.
            let conflicts_b = storage_conflicts.count(local_b);
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
        let mut used_variants = DenseBitSet::new_empty(variant_fields.len());
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
            assignments[local] = Ineligible(Some(FieldIdx::new(idx)));
        }
    }
    debug!("coroutine saved local assignments: {:?}", assignments);

    (ineligible_locals, assignments)
}

/// Compute the full coroutine layout.
pub(super) fn layout<
    'a,
    F: core::ops::Deref<Target = &'a LayoutData<FieldIdx, VariantIdx>> + core::fmt::Debug + Copy,
    VariantIdx: Idx,
    FieldIdx: Idx,
    LocalIdx: Idx,
>(
    calc: &super::LayoutCalculator<impl HasDataLayout>,
    local_layouts: &IndexSlice<LocalIdx, F>,
    mut prefix_layouts: IndexVec<FieldIdx, F>,
    variant_fields: &IndexSlice<VariantIdx, IndexVec<FieldIdx, LocalIdx>>,
    storage_conflicts: &BitMatrix<LocalIdx, LocalIdx>,
    tag_to_layout: impl Fn(Scalar) -> F,
) -> super::LayoutCalculatorResult<FieldIdx, VariantIdx, F> {
    use SavedLocalEligibility::*;

    let (ineligible_locals, assignments) =
        coroutine_saved_local_eligibility(local_layouts.len(), variant_fields, storage_conflicts);

    // Build a prefix layout, including "promoting" all ineligible
    // locals as part of the prefix. We compute the layout of all of
    // these fields at once to get optimal packing.
    let tag_index = prefix_layouts.next_index();

    // `variant_fields` already accounts for the reserved variants, so no need to add them.
    let max_discr = (variant_fields.len() - 1) as u128;
    let discr_int = Integer::fit_unsigned(max_discr);
    let tag = Scalar::Initialized {
        value: Primitive::Int(discr_int, /* signed = */ false),
        valid_range: WrappingRange { start: 0, end: max_discr },
    };

    let promoted_layouts = ineligible_locals.iter().map(|local| local_layouts[local]);
    prefix_layouts.push(tag_to_layout(tag));
    prefix_layouts.extend(promoted_layouts);
    let prefix =
        calc.univariant(&prefix_layouts, &ReprOptions::default(), StructKind::AlwaysSized)?;

    let (prefix_size, prefix_align) = (prefix.size, prefix.align);

    // Split the prefix layout into the "outer" fields (upvars and
    // discriminant) and the "promoted" fields. Promoted fields will
    // get included in each variant that requested them in
    // CoroutineLayout.
    debug!("prefix = {:#?}", prefix);
    let (outer_fields, promoted_offsets, promoted_memory_index) = match prefix.fields {
        FieldsShape::Arbitrary { mut offsets, memory_index } => {
            let mut inverse_memory_index = memory_index.invert_bijective_mapping();

            // "a" (`0..b_start`) and "b" (`b_start..`) correspond to
            // "outer" and "promoted" fields respectively.
            let b_start = tag_index.plus(1);
            let offsets_b = IndexVec::from_raw(offsets.raw.split_off(b_start.index()));
            let offsets_a = offsets;

            // Disentangle the "a" and "b" components of `inverse_memory_index`
            // by preserving the order but keeping only one disjoint "half" each.
            // FIXME(eddyb) build a better abstraction for permutations, if possible.
            let inverse_memory_index_b: IndexVec<u32, FieldIdx> = inverse_memory_index
                .iter()
                .filter_map(|&i| i.index().checked_sub(b_start.index()).map(FieldIdx::new))
                .collect();
            inverse_memory_index.raw.retain(|&i| i.index() < b_start.index());
            let inverse_memory_index_a = inverse_memory_index;

            // Since `inverse_memory_index_{a,b}` each only refer to their
            // respective fields, they can be safely inverted
            let memory_index_a = inverse_memory_index_a.invert_bijective_mapping();
            let memory_index_b = inverse_memory_index_b.invert_bijective_mapping();

            let outer_fields =
                FieldsShape::Arbitrary { offsets: offsets_a, memory_index: memory_index_a };
            (outer_fields, offsets_b, memory_index_b)
        }
        _ => unreachable!(),
    };

    let mut size = prefix.size;
    let mut align = prefix.align;
    let variants = variant_fields
        .iter_enumerated()
        .map(|(index, variant_fields)| {
            // Only include overlap-eligible fields when we compute our variant layout.
            let variant_only_tys = variant_fields
                .iter()
                .filter(|local| match assignments[**local] {
                    Unassigned => unreachable!(),
                    Assigned(v) if v == index => true,
                    Assigned(_) => unreachable!("assignment does not match variant"),
                    Ineligible(_) => false,
                })
                .map(|local| local_layouts[*local]);

            let mut variant = calc.univariant(
                &variant_only_tys.collect::<IndexVec<_, _>>(),
                &ReprOptions::default(),
                StructKind::Prefixed(prefix_size, prefix_align.abi),
            )?;
            variant.variants = Variants::Single { index };

            let FieldsShape::Arbitrary { offsets, memory_index } = variant.fields else {
                unreachable!();
            };

            // Now, stitch the promoted and variant-only fields back together in
            // the order they are mentioned by our CoroutineLayout.
            // Because we only use some subset (that can differ between variants)
            // of the promoted fields, we can't just pick those elements of the
            // `promoted_memory_index` (as we'd end up with gaps).
            // So instead, we build an "inverse memory_index", as if all of the
            // promoted fields were being used, but leave the elements not in the
            // subset as `invalid_field_idx`, which we can filter out later to
            // obtain a valid (bijective) mapping.
            let invalid_field_idx = promoted_memory_index.len() + memory_index.len();
            let mut combined_inverse_memory_index =
                IndexVec::from_elem_n(FieldIdx::new(invalid_field_idx), invalid_field_idx);

            let mut offsets_and_memory_index = iter::zip(offsets, memory_index);
            let combined_offsets = variant_fields
                .iter_enumerated()
                .map(|(i, local)| {
                    let (offset, memory_index) = match assignments[*local] {
                        Unassigned => unreachable!(),
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
            combined_inverse_memory_index.raw.retain(|&i| i.index() != invalid_field_idx);
            let combined_memory_index = combined_inverse_memory_index.invert_bijective_mapping();

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

    let uninhabited = prefix.uninhabited || variants.iter().all(|v| v.is_uninhabited());
    let abi = BackendRepr::Memory { sized: true };

    Ok(LayoutData {
        variants: Variants::Multiple {
            tag,
            tag_encoding: TagEncoding::Direct,
            tag_field: tag_index,
            variants,
        },
        fields: outer_fields,
        backend_repr: abi,
        // Suppress niches inside coroutines. If the niche is inside a field that is aliased (due to
        // self-referentiality), getting the discriminant can cause aliasing violations.
        // `UnsafeCell` blocks niches for the same reason, but we don't yet have `UnsafePinned` that
        // would do the same for us here.
        // See <https://github.com/rust-lang/rust/issues/63818>, <https://github.com/rust-lang/miri/issues/3780>.
        // FIXME: Remove when <https://github.com/rust-lang/rust/issues/125735> is implemented and aliased coroutine fields are wrapped in `UnsafePinned`.
        largest_niche: None,
        uninhabited,
        size,
        align,
        max_repr_align: None,
        unadjusted_abi_align: align.abi,
        randomization_seed: Default::default(),
    })
}
