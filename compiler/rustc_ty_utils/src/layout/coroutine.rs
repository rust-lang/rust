use std::cmp::Reverse;
use std::collections::BTreeSet;
use std::fmt::Debug;
use std::ops::Bound;

use rustc_abi::{
    BackendRepr, FieldIdx, FieldsShape, Integer, Layout, LayoutData, Primitive, Scalar, Size,
    TagEncoding, TyAndLayout, Variants, WrappingRange,
};
use rustc_index::bit_set::BitSet;
use rustc_index::{Idx, IndexVec};
use rustc_middle::mir::CoroutineSavedLocal;
use rustc_middle::ty::layout::{HasTyCtxt, IntegerExt, LayoutCx, LayoutError, LayoutOf};
use rustc_middle::ty::{EarlyBinder, GenericArgsRef, Ty};
use tracing::{debug, instrument};

use super::error;

#[instrument(level = "debug", skip(cx))]
pub(super) fn coroutine_layout<'tcx>(
    cx: &LayoutCx<'tcx>,
    ty: Ty<'tcx>,
    def_id: rustc_hir::def_id::DefId,
    args: GenericArgsRef<'tcx>,
) -> Result<Layout<'tcx>, &'tcx LayoutError<'tcx>> {
    let tcx = cx.tcx();
    let Some(info) = tcx.coroutine_layout(def_id, args.as_coroutine().kind_ty()) else {
        return Err(error(cx, LayoutError::Unknown(ty)));
    };

    let tcx = cx.tcx();
    let field_layouts: IndexVec<CoroutineSavedLocal, _> = info
        .field_tys
        .iter_enumerated()
        .map(|(saved_local, ty)| {
            let ty = EarlyBinder::bind(ty.ty).instantiate(tcx, args);
            cx.spanned_layout_of(ty, info.field_tys[saved_local].source_info.span)
        })
        .try_collect()?;
    let layouts: IndexVec<CoroutineSavedLocal, _> =
        field_layouts.iter().map(|data| data.layout.clone()).collect();

    let field_sort_keys: IndexVec<CoroutineSavedLocal, _>;
    let mut saved_locals: Vec<_>;
    // ## The heuristic on which saved locals get allocation first ##
    // 1. the alignment
    // Intuitively data with a larger alignment asks for a larger block of contiguous memory.
    // It is easier to get large blocks early in the beginning, but it will get harder to
    // recover them as fragmentation creeps in when data with smaller alignment occupies
    // the large chunks.
    // 2. the size
    // The size also poses restriction on layout, but not as potent as alignment.
    // 3. the degree of conflicts
    // This metric is the number of confliciting saved locals with a given saved local.
    // Preferring allocating highly conflicting data over those that are less and more
    // transient in nature will keep the fragmentation contained in neighbourhoods of a layout.
    (saved_locals, field_sort_keys) = field_layouts
        .iter_enumerated()
        .map(|(saved_local, ty)| {
            (
                saved_local,
                (
                    Reverse(ty.align.abi),
                    Reverse(ty.size),
                    Reverse(info.storage_conflicts.count(saved_local)),
                ),
            )
        })
        .unzip();
    let mut uninhabited_or_zst = BitSet::new_empty(field_layouts.len());
    for (saved_local, ty) in field_layouts.iter_enumerated() {
        if ty.backend_repr.is_uninhabited() || ty.is_zst() {
            uninhabited_or_zst.insert(saved_local);
        }
    }
    saved_locals.sort_by_key(|&idx| &field_sort_keys[idx]);
    let max_discr = (info.variant_fields.len() - 1) as u128;
    let discr_int = Integer::fit_unsigned(max_discr);
    let tag = Scalar::Initialized {
        value: Primitive::Int(discr_int, false),
        valid_range: WrappingRange { start: 0, end: max_discr },
    };
    let tag_layout = TyAndLayout {
        ty: discr_int.to_ty(tcx, false),
        layout: tcx.mk_layout(LayoutData::scalar(cx, tag)),
    };
    // This will be *the* align of the entire coroutine,
    // which is the maximal alignment of all saved locals.
    // We need to also consider the tag layout alignment.
    let align = saved_locals
        .get(0)
        .map(|&idx| layouts[idx].align.max(tag_layout.align))
        .unwrap_or(tag_layout.align);

    // ## The blocked map, or the reservation map ##
    // This map from saved locals to memory layout records the reservation
    // status of the coroutine state memory, down to the byte granularity.
    // `Slot`s are inserted to mark ranges of memory that a particular saved local
    // shall not have overlapping memory allocation, due to the liveness of
    // other conflicting saved locals.
    // Therefore, we can try to make reservation for this saved local
    // by inspecting the gaps before, between, and after those blocked-out memory ranges.
    let mut blocked: IndexVec<CoroutineSavedLocal, BTreeSet<Slot>> =
        IndexVec::from_elem_n(BTreeSet::new(), saved_locals.len());
    let mut tag_blocked = BTreeSet::new();
    let mut assignment = IndexVec::from_elem_n(Slot { start: 0, end: 0 }, saved_locals.len());
    for (idx, &current_local) in saved_locals.iter().enumerate() {
        if uninhabited_or_zst.contains(current_local) {
            // Do not bother to compute on uninhabited data.
            // They will not get allocation after all.
            // By default, a ZST occupies the beginning of the coroutine state.
            continue;
        }
        let layout_data = &field_layouts[current_local];

        let candidate = find_lowest_viable_allocation(&layout_data, &blocked[current_local]);
        // The discriminant is certainly conflicting with all the saved locals
        merge_slot_in(&mut tag_blocked, candidate);
        for &other_local in &saved_locals[idx + 1..] {
            if info.storage_conflicts.contains(current_local, other_local) {
                merge_slot_in(&mut blocked[other_local], candidate);
            }
        }
        // Adjustment to the layout of this field by shifting them into the chosen slot
        assignment[current_local] = candidate;
    }
    debug!(assignment = ?assignment.debug_map_view());

    // Find a slot for discriminant, also known as the tag.
    let tag_candidate = find_lowest_viable_allocation(&tag_layout, &tag_blocked);
    debug!(tag = ?tag_candidate);

    // Assemble the layout for each coroutine state
    let variants: IndexVec<_, LayoutData<_, _>> = info
        .variant_fields
        .iter_enumerated()
        .map(|(index, fields)| {
            let size = Size::from_bytes(
                fields
                    .iter()
                    .map(|&saved_local| assignment[saved_local].end)
                    .max()
                    .unwrap_or(0)
                    .max(tag_candidate.end),
            )
            .align_to(align.abi);
            let offsets: IndexVec<_, _> = fields
                .iter()
                .map(|&saved_local| Size::from_bytes(assignment[saved_local].start))
                .collect();
            let memory_index = IndexVec::from_fn_n(|n: FieldIdx| (n.index() as u32), offsets.len());
            LayoutData {
                // We are aware of specialized layouts such as scalar pairs but this is still
                // in development.
                // Let us hold off from further optimisation until more information is available.
                fields: FieldsShape::Arbitrary { offsets, memory_index },
                variants: Variants::Single { index },
                backend_repr: BackendRepr::Memory { sized: true },
                largest_niche: None,
                align,
                size,
                max_repr_align: None,
                unadjusted_abi_align: align.abi,
            }
        })
        .collect();
    let size = variants
        .iter()
        .map(|data| data.size)
        .max()
        .unwrap_or(Size::ZERO)
        .max(Size::from_bytes(tag_candidate.end))
        .align_to(align.abi);
    let layout = tcx.mk_layout(LayoutData {
        fields: FieldsShape::Arbitrary {
            offsets: [Size::from_bytes(tag_candidate.start)].into(),
            memory_index: [0].into(),
        },
        variants: Variants::Multiple {
            tag,
            tag_encoding: TagEncoding::Direct,
            tag_field: 0,
            variants,
        },
        backend_repr: BackendRepr::Memory { sized: true },
        // Suppress niches inside coroutines. If the niche is inside a field that is aliased (due to
        // self-referentiality), getting the discriminant can cause aliasing violations.
        // `UnsafeCell` blocks niches for the same reason, but we don't yet have `UnsafePinned` that
        // would do the same for us here.
        // See <https://github.com/rust-lang/rust/issues/63818>, <https://github.com/rust-lang/miri/issues/3780>.
        // FIXME(#125735): Remove when <https://github.com/rust-lang/rust/issues/125735>
        // is implemented and aliased coroutine fields are wrapped in `UnsafePinned`.
        // NOTE(@dingxiangfei2009): I believe there is still niche, which is the tag,
        // but I am not sure how much benefit is there for us to grab.
        largest_niche: None,
        align,
        size,
        max_repr_align: None,
        unadjusted_abi_align: align.abi,
    });
    debug!("coroutine layout ({:?}): {:#?}", ty, layout);
    Ok(layout)
}

/// An occupied slot in the coroutine memory at some yield point
#[derive(PartialEq, Eq, Copy, Clone)]
struct Slot {
    /// Beginning of the memory slot, inclusive
    start: u64,
    /// End of the memory slot, exclusive or one byte past the data
    end: u64,
}

impl PartialOrd for Slot {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (self.start, self.end).partial_cmp(&(other.start, other.end))
    }
}

impl Ord for Slot {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.start, self.end).cmp(&(other.start, other.end))
    }
}

impl Debug for Slot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Slot").field(&self.start).field(&self.end).finish()
    }
}

impl Slot {
    fn overlap_with(&self, other: &Self) -> bool {
        if self.start == self.end || other.start == other.end {
            return false;
        }
        self.contains_point(other.start) || other.contains_point(self.start)
    }
    fn contains_point(&self, point: u64) -> bool {
        self.start <= point && point < self.end
    }
}

fn merge_slot_in(slots: &mut BTreeSet<Slot>, slot: Slot) {
    let start = Slot { start: slot.start, end: slot.start };
    let end = Slot { start: slot.end, end: slot.end };
    let one_past_end = Slot { start: slot.end + 1, end: slot.end + 1 };
    let (range_start, replace_start) = if let Some(prev) = slots.range(..start).next_back()
        && (prev.end == slot.start || prev.contains_point(slot.start))
    {
        (Bound::Included(prev), prev.start)
    } else {
        (Bound::Included(&start), slot.start)
    };
    let (range_end, replace_end) = if let Some(next) = slots.range(..one_past_end).next_back()
        && next.start == slot.end
    {
        (Bound::Included(next), next.end)
    } else if let Some(prev) = slots.range(..end).next_back()
        && prev.contains_point(slot.end)
    {
        (Bound::Included(prev), prev.end)
    } else {
        (Bound::Included(&end), slot.end)
    };
    let to_remove: Vec<_> = slots.range((range_start, range_end)).copied().collect();
    for slot in to_remove {
        slots.remove(&slot);
    }
    slots.insert(Slot { start: replace_start, end: replace_end });
}

fn find_lowest_viable_allocation<F: Idx, V: Idx>(
    layout: &LayoutData<F, V>,
    blocked: &BTreeSet<Slot>,
) -> Slot {
    let size = layout.size.bytes();
    let align = layout.align.abi;
    let mut candidate = Slot { start: 0, end: size };
    for slot in blocked {
        if slot.overlap_with(&candidate) {
            let start = Size::from_bytes(slot.end).align_to(align).bytes();
            candidate = Slot { start, end: start + size };
        } else {
            break;
        }
    }
    candidate
}
