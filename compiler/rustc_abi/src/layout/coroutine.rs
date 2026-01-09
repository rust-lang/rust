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
    Align, BackendRepr, FieldsShape, HasDataLayout, Integer, LayoutData, Primitive, ReprOptions,
    Scalar, StructKind, TagEncoding, Variants, WrappingRange,
};

/// This option controls how coroutine saved locals are packed
/// into the coroutine state data
#[derive(Debug, Clone, Copy)]
pub enum PackCoroutineLayout {
    /// The classic layout where captures are always promoted to coroutine state prefix
    Classic,
    /// Captures are first saved into the `UNRESUME` state and promoted
    /// when they are used across more than one suspension
    CapturesOnly,
}

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
    debug!(?ineligible_locals, "after counting variants containing a saved local");

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
    debug!(?ineligible_locals, "after checking conflicts");

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
        debug!(?ineligible_locals, "after checking used variants");
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
    relocated_upvars: &IndexSlice<LocalIdx, Option<LocalIdx>>,
    upvar_layouts: IndexVec<FieldIdx, F>,
    variant_fields: &IndexSlice<VariantIdx, IndexVec<FieldIdx, LocalIdx>>,
    storage_conflicts: &BitMatrix<LocalIdx, LocalIdx>,
    pack: PackCoroutineLayout,
    tag_to_layout: impl Fn(Scalar) -> F,
) -> super::LayoutCalculatorResult<FieldIdx, VariantIdx, F> {
    use SavedLocalEligibility::*;

    let (ineligible_locals, assignments) =
        coroutine_saved_local_eligibility(local_layouts.len(), variant_fields, storage_conflicts);
    debug!(?ineligible_locals);

    // Build a prefix layout, consisting of only the state tag and, as per request, upvars
    let tag_index = match pack {
        PackCoroutineLayout::CapturesOnly => FieldIdx::new(0),
        PackCoroutineLayout::Classic => upvar_layouts.next_index(),
    };

    // `variant_fields` already accounts for the reserved variants, so no need to add them.
    let max_discr = (variant_fields.len() - 1) as u128;
    let discr_int = Integer::fit_unsigned(max_discr);
    let tag = Scalar::Initialized {
        value: Primitive::Int(discr_int, /* signed = */ false),
        valid_range: WrappingRange { start: 0, end: max_discr },
    };

    let upvars_in_unresumed: rustc_hash::FxHashSet<_> =
        variant_fields[VariantIdx::new(0)].iter().copied().collect();
    let promoted_layouts = ineligible_locals.iter().filter_map(|local| {
        if matches!(pack, PackCoroutineLayout::Classic) && upvars_in_unresumed.contains(&local) {
            // We do not need to promote upvars, they are already in the upvar region
            None
        } else {
            Some(local_layouts[local])
        }
    });
    // FIXME: when we introduce more pack scheme, we need to change the prefix layout here
    let prefix_layouts: IndexVec<_, _> = match pack {
        PackCoroutineLayout::Classic => {
            // Classic scheme packs the states as follows
            // [ <upvars>.. , <state tag>, <promoted ineligibles>] ++ <variant data>
            // In addition, UNRESUME overlaps with the <upvars> part
            upvar_layouts.into_iter().chain([tag_to_layout(tag)]).chain(promoted_layouts).collect()
        }
        PackCoroutineLayout::CapturesOnly => {
            [tag_to_layout(tag)].into_iter().chain(promoted_layouts).collect()
        }
    };
    debug!(?pack, "prefix_layouts={prefix_layouts:#?}");
    let prefix =
        calc.univariant(&prefix_layouts, &ReprOptions::default(), StructKind::AlwaysSized)?;

    let prefix_size = prefix.size;

    // Split the prefix layout into the discriminant and
    // the "promoted" fields.
    // Promoted fields will get included in each variant
    // that requested them in CoroutineLayout.
    debug!("prefix={prefix:#?}");
    let (outer_fields, promoted_offsets, promoted_memory_index) = match prefix.fields {
        FieldsShape::Arbitrary { mut offsets, in_memory_order } => {
            // "a" (`0..b_start`) and "b" (`b_start..`) correspond to
            // "outer" and "promoted" fields respectively.
            let b_start = tag_index.plus(1);
            let offsets_b = IndexVec::from_raw(offsets.raw.split_off(b_start.index()));
            let offsets_a = offsets;

            // Disentangle the "a" and "b" components of `in_memory_order`
            // by preserving the order but keeping only one disjoint "half" each.
            // FIXME(eddyb) build a better abstraction for permutations, if possible.
            let mut in_memory_order_a = IndexVec::<u32, FieldIdx>::new();
            let mut in_memory_order_b = IndexVec::<u32, FieldIdx>::new();
            for i in in_memory_order {
                if let Some(j) = i.index().checked_sub(b_start.index()) {
                    in_memory_order_b.push(FieldIdx::new(j));
                } else {
                    in_memory_order_a.push(i);
                }
            }

            let outer_fields =
                FieldsShape::Arbitrary { offsets: offsets_a, in_memory_order: in_memory_order_a };
            (outer_fields, offsets_b, in_memory_order_b.invert_bijective_mapping())
        }
        _ => unreachable!(),
    };

    // Here we start to compute layout of each state variant
    let mut size = prefix.size;
    let mut align = prefix.align;
    let variants = variant_fields
        .iter_enumerated()
        .map(|(index, variant_fields)| {
            // Special case: UNRESUMED overlaps with the upvar region of the prefix,
            // so that moving upvars may eventually become a no-op.
            let is_unresumed = index.index() == 0;
            if is_unresumed && matches!(pack, PackCoroutineLayout::Classic) {
                let fields = FieldsShape::Arbitrary {
                    offsets: (0..tag_index.index()).map(|i| outer_fields.offset(i)).collect(),
                    in_memory_order: (0..tag_index.index()).map(FieldIdx::new).collect(),
                };
                let align = prefix.align;
                let size = prefix.size;
                return Ok(LayoutData {
                    fields,
                    variants: Variants::Single { index },
                    backend_repr: BackendRepr::Memory { sized: true },
                    largest_niche: None,
                    uninhabited: false,
                    align,
                    size,
                    max_repr_align: None,
                    unadjusted_abi_align: align.abi,
                    randomization_seed: Default::default(),
                });
            }
            let mut is_ineligible = IndexVec::from_elem_n(None, variant_fields.len());
            for (field, &local) in variant_fields.iter_enumerated() {
                if is_unresumed {
                    if let Some(inner_local) = relocated_upvars[local]
                        && inner_local != local
                        && let Ineligible(Some(promoted_field)) = assignments[inner_local]
                    {
                        is_ineligible.insert(field, promoted_field);
                        continue;
                    }
                }
                match assignments[local] {
                    Assigned(v) if v == index => {}
                    Ineligible(Some(promoted_field)) => {
                        is_ineligible.insert(field, promoted_field);
                    }
                    Ineligible(None) => {
                        panic!("an ineligible local should have been promoted into the prefix")
                    }
                    Assigned(_) => {
                        panic!("an eligible local should have been assigned to exactly one variant")
                    }
                    Unassigned => {
                        panic!("each saved local should have been inspected at least once")
                    }
                }
            }
            // Only include overlap-eligible fields when we compute our variant layout.
            let fields: IndexVec<_, _> = variant_fields
                .iter_enumerated()
                .filter_map(|(field, &local)| {
                    if is_ineligible.contains(field) { None } else { Some(local_layouts[local]) }
                })
                .collect();

            let mut variant = calc.univariant(
                &fields,
                &ReprOptions::default(),
                StructKind::Prefixed(prefix_size, Align::ONE),
            )?;
            variant.variants = Variants::Single { index };

            let FieldsShape::Arbitrary { offsets, in_memory_order } = variant.fields else {
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
            let memory_index = in_memory_order.invert_bijective_mapping();
            let invalid_field_idx = promoted_memory_index.len() + memory_index.len();
            let mut combined_in_memory_order =
                IndexVec::from_elem_n(FieldIdx::new(invalid_field_idx), invalid_field_idx);

            let mut offsets_and_memory_index = iter::zip(offsets, memory_index);
            let combined_offsets = is_ineligible
                .iter_enumerated()
                .map(|(i, &is_ineligible)| {
                    let (offset, memory_index) = if let Some(field_idx) = is_ineligible {
                        (promoted_offsets[field_idx], promoted_memory_index[field_idx])
                    } else {
                        let (offset, memory_index) = offsets_and_memory_index.next().unwrap();
                        (offset, promoted_memory_index.len() as u32 + memory_index)
                    };
                    combined_in_memory_order[memory_index] = i;
                    offset
                })
                .collect();

            // Remove the unused slots to obtain the combined `in_memory_order`
            // (also see previous comment).
            combined_in_memory_order.raw.retain(|&i| i.index() != invalid_field_idx);

            variant.fields = FieldsShape::Arbitrary {
                offsets: combined_offsets,
                in_memory_order: combined_in_memory_order,
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
