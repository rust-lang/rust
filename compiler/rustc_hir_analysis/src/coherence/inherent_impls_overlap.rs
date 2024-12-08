use rustc_data_structures::fx::{FxIndexMap, FxIndexSet, IndexEntry};
use rustc_errors::codes::*;
use rustc_errors::struct_span_code_err;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_index::IndexVec;
use rustc_middle::traits::specialization_graph::OverlapMode;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::{ErrorGuaranteed, Symbol};
use rustc_trait_selection::traits::{self, SkipLeakCheck};
use smallvec::SmallVec;
use tracing::debug;

pub(crate) fn crate_inherent_impls_overlap_check(
    tcx: TyCtxt<'_>,
    (): (),
) -> Result<(), ErrorGuaranteed> {
    let mut inherent_overlap_checker = InherentOverlapChecker { tcx };
    let mut res = Ok(());
    for id in tcx.hir().items() {
        res = res.and(inherent_overlap_checker.check_item(id));
    }
    res
}

struct InherentOverlapChecker<'tcx> {
    tcx: TyCtxt<'tcx>,
}

rustc_index::newtype_index! {
    #[orderable]
    pub struct RegionId {}
}

impl<'tcx> InherentOverlapChecker<'tcx> {
    /// Checks whether any associated items in impls 1 and 2 share the same identifier and
    /// namespace.
    fn impls_have_common_items(
        &self,
        impl_items1: &ty::AssocItems,
        impl_items2: &ty::AssocItems,
    ) -> bool {
        let mut impl_items1 = &impl_items1;
        let mut impl_items2 = &impl_items2;

        // Performance optimization: iterate over the smaller list
        if impl_items1.len() > impl_items2.len() {
            std::mem::swap(&mut impl_items1, &mut impl_items2);
        }

        for &item1 in impl_items1.in_definition_order() {
            let collision = impl_items2
                .filter_by_name_unhygienic(item1.name)
                .any(|&item2| self.compare_hygienically(item1, item2));

            if collision {
                return true;
            }
        }

        false
    }

    fn compare_hygienically(&self, item1: ty::AssocItem, item2: ty::AssocItem) -> bool {
        // Symbols and namespace match, compare hygienically.
        item1.kind.namespace() == item2.kind.namespace()
            && item1.ident(self.tcx).normalize_to_macros_2_0()
                == item2.ident(self.tcx).normalize_to_macros_2_0()
    }

    fn check_for_duplicate_items_in_impl(&self, impl_: DefId) -> Result<(), ErrorGuaranteed> {
        let impl_items = self.tcx.associated_items(impl_);

        let mut seen_items = FxIndexMap::default();
        let mut res = Ok(());
        for impl_item in impl_items.in_definition_order() {
            let span = self.tcx.def_span(impl_item.def_id);
            let ident = impl_item.ident(self.tcx);

            let norm_ident = ident.normalize_to_macros_2_0();
            match seen_items.entry(norm_ident) {
                IndexEntry::Occupied(entry) => {
                    let former = entry.get();
                    res = Err(struct_span_code_err!(
                        self.tcx.dcx(),
                        span,
                        E0592,
                        "duplicate definitions with name `{}`",
                        ident,
                    )
                    .with_span_label(span, format!("duplicate definitions for `{ident}`"))
                    .with_span_label(*former, format!("other definition for `{ident}`"))
                    .emit());
                }
                IndexEntry::Vacant(entry) => {
                    entry.insert(span);
                }
            }
        }
        res
    }

    fn check_for_common_items_in_impls(
        &self,
        impl1: DefId,
        impl2: DefId,
        overlap: traits::OverlapResult<'_>,
    ) -> Result<(), ErrorGuaranteed> {
        let impl_items1 = self.tcx.associated_items(impl1);
        let impl_items2 = self.tcx.associated_items(impl2);

        let mut res = Ok(());
        for &item1 in impl_items1.in_definition_order() {
            let collision = impl_items2
                .filter_by_name_unhygienic(item1.name)
                .find(|&&item2| self.compare_hygienically(item1, item2));

            if let Some(item2) = collision {
                let name = item1.ident(self.tcx).normalize_to_macros_2_0();
                let mut err = struct_span_code_err!(
                    self.tcx.dcx(),
                    self.tcx.def_span(item1.def_id),
                    E0592,
                    "duplicate definitions with name `{}`",
                    name
                );
                err.span_label(
                    self.tcx.def_span(item1.def_id),
                    format!("duplicate definitions for `{name}`"),
                );
                err.span_label(
                    self.tcx.def_span(item2.def_id),
                    format!("other definition for `{name}`"),
                );

                for cause in &overlap.intercrate_ambiguity_causes {
                    cause.add_intercrate_ambiguity_hint(&mut err);
                }

                if overlap.involves_placeholder {
                    traits::add_placeholder_note(&mut err);
                }

                res = Err(err.emit());
            }
        }
        res
    }

    fn check_for_overlapping_inherent_impls(
        &self,
        overlap_mode: OverlapMode,
        impl1_def_id: DefId,
        impl2_def_id: DefId,
    ) -> Result<(), ErrorGuaranteed> {
        let maybe_overlap = traits::overlapping_impls(
            self.tcx,
            impl1_def_id,
            impl2_def_id,
            // We go ahead and just skip the leak check for
            // inherent impls without warning.
            SkipLeakCheck::Yes,
            overlap_mode,
        );

        if let Some(overlap) = maybe_overlap {
            self.check_for_common_items_in_impls(impl1_def_id, impl2_def_id, overlap)
        } else {
            Ok(())
        }
    }

    fn check_item(&mut self, id: hir::ItemId) -> Result<(), ErrorGuaranteed> {
        let def_kind = self.tcx.def_kind(id.owner_id);
        if !matches!(def_kind, DefKind::Enum | DefKind::Struct | DefKind::Trait | DefKind::Union) {
            return Ok(());
        }

        let impls = self.tcx.inherent_impls(id.owner_id);
        let overlap_mode = OverlapMode::get(self.tcx, id.owner_id.to_def_id());

        let impls_items = impls
            .iter()
            .map(|impl_def_id| (impl_def_id, self.tcx.associated_items(*impl_def_id)))
            .collect::<SmallVec<[_; 8]>>();

        // Perform a O(n^2) algorithm for small n,
        // otherwise switch to an allocating algorithm with
        // faster asymptotic runtime.
        const ALLOCATING_ALGO_THRESHOLD: usize = 500;
        let mut res = Ok(());
        if impls.len() < ALLOCATING_ALGO_THRESHOLD {
            for (i, &(&impl1_def_id, impl_items1)) in impls_items.iter().enumerate() {
                res = res.and(self.check_for_duplicate_items_in_impl(impl1_def_id));

                for &(&impl2_def_id, impl_items2) in &impls_items[(i + 1)..] {
                    if self.impls_have_common_items(impl_items1, impl_items2) {
                        res = res.and(self.check_for_overlapping_inherent_impls(
                            overlap_mode,
                            impl1_def_id,
                            impl2_def_id,
                        ));
                    }
                }
            }
        } else {
            // Build a set of connected regions of impl blocks.
            // Two impl blocks are regarded as connected if they share
            // an item with the same unhygienic identifier.
            // After we have assembled the connected regions,
            // run the O(n^2) algorithm on each connected region.
            // This is advantageous to running the algorithm over the
            // entire graph when there are many connected regions.

            struct ConnectedRegion {
                idents: SmallVec<[Symbol; 8]>,
                impl_blocks: FxIndexSet<usize>,
            }
            let mut connected_regions: IndexVec<RegionId, _> = Default::default();
            // Reverse map from the Symbol to the connected region id.
            let mut connected_region_ids = FxIndexMap::default();

            for (i, &(&_impl_def_id, impl_items)) in impls_items.iter().enumerate() {
                if impl_items.len() == 0 {
                    continue;
                }
                // First obtain a list of existing connected region ids
                let mut idents_to_add = SmallVec::<[Symbol; 8]>::new();
                let mut ids = impl_items
                    .in_definition_order()
                    .filter_map(|item| {
                        let entry = connected_region_ids.entry(item.name);
                        if let IndexEntry::Occupied(e) = &entry {
                            Some(*e.get())
                        } else {
                            idents_to_add.push(item.name);
                            None
                        }
                    })
                    .collect::<SmallVec<[RegionId; 8]>>();
                // Sort the id list so that the algorithm is deterministic
                ids.sort_unstable();
                ids.dedup();
                let ids = ids;
                match &ids[..] {
                    // Create a new connected region
                    [] => {
                        let id_to_set = connected_regions.next_index();
                        // Update the connected region ids
                        for ident in &idents_to_add {
                            connected_region_ids.insert(*ident, id_to_set);
                        }
                        connected_regions.insert(id_to_set, ConnectedRegion {
                            idents: idents_to_add,
                            impl_blocks: std::iter::once(i).collect(),
                        });
                    }
                    // Take the only id inside the list
                    &[id_to_set] => {
                        let region = connected_regions[id_to_set].as_mut().unwrap();
                        region.impl_blocks.insert(i);
                        region.idents.extend_from_slice(&idents_to_add);
                        // Update the connected region ids
                        for ident in &idents_to_add {
                            connected_region_ids.insert(*ident, id_to_set);
                        }
                    }
                    // We have multiple connected regions to merge.
                    // In the worst case this might add impl blocks
                    // one by one and can thus be O(n^2) in the size
                    // of the resulting final connected region, but
                    // this is no issue as the final step to check
                    // for overlaps runs in O(n^2) as well.
                    &[id_to_set, ..] => {
                        let mut region = connected_regions.remove(id_to_set).unwrap();
                        region.impl_blocks.insert(i);
                        region.idents.extend_from_slice(&idents_to_add);
                        // Update the connected region ids
                        for ident in &idents_to_add {
                            connected_region_ids.insert(*ident, id_to_set);
                        }

                        // Remove other regions from ids.
                        for &id in ids.iter() {
                            if id == id_to_set {
                                continue;
                            }
                            let r = connected_regions.remove(id).unwrap();
                            for ident in r.idents.iter() {
                                connected_region_ids.insert(*ident, id_to_set);
                            }
                            region.idents.extend_from_slice(&r.idents);
                            region.impl_blocks.extend(r.impl_blocks);
                        }

                        connected_regions.insert(id_to_set, region);
                    }
                }
            }

            debug!(
                "churning through {} components (sum={}, avg={}, var={}, max={})",
                connected_regions.len(),
                impls.len(),
                impls.len() / connected_regions.len(),
                {
                    let avg = impls.len() / connected_regions.len();
                    let s = connected_regions
                        .iter()
                        .flatten()
                        .map(|r| r.impl_blocks.len() as isize - avg as isize)
                        .map(|v| v.unsigned_abs())
                        .sum::<usize>();
                    s / connected_regions.len()
                },
                connected_regions.iter().flatten().map(|r| r.impl_blocks.len()).max().unwrap()
            );
            // List of connected regions is built. Now, run the overlap check
            // for each pair of impl blocks in the same connected region.
            for region in connected_regions.into_iter().flatten() {
                let impl_blocks = region.impl_blocks.into_iter().collect::<SmallVec<[usize; 8]>>();
                for (i, &impl1_items_idx) in impl_blocks.iter().enumerate() {
                    let &(&impl1_def_id, impl_items1) = &impls_items[impl1_items_idx];
                    res = res.and(self.check_for_duplicate_items_in_impl(impl1_def_id));

                    for &impl2_items_idx in impl_blocks[(i + 1)..].iter() {
                        let &(&impl2_def_id, impl_items2) = &impls_items[impl2_items_idx];
                        if self.impls_have_common_items(impl_items1, impl_items2) {
                            res = res.and(self.check_for_overlapping_inherent_impls(
                                overlap_mode,
                                impl1_def_id,
                                impl2_def_id,
                            ));
                        }
                    }
                }
            }
        }
        res
    }
}
