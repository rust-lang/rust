use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap, FxIndexSet};
use rustc_data_structures::unord::UnordMap;
use rustc_hir::def_id::DefId;
use rustc_index::Idx;
use rustc_index::bit_set::{DenseBitSet, SparseBitMatrix};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, RegionVid, TyCtxt, TypeVisitable, TypeVisitor};

use crate::constraints::ConstraintSccIndex;
use crate::region_infer::RegionInferenceContext;

/// TypeVisitor that walks a generic arg tree in depth-first order, assigning a
/// walk-order index to every region encountered. Only `ReVar` regions produce
/// entries in the output mapping; other region kinds (e.g. `ReBound` from
/// higher-ranked types) advance the index without producing entries, keeping
/// positions aligned with a consumer's walk over the same structure after
/// monomorphization.
struct RegionIndexCollector<'a, 'tcx> {
    walk_index: usize,
    mappings: &'a mut UnordMap<u32, u32>,
    relevant_vids: &'a mut FxIndexSet<RegionVid>,
    external_region_to_vid: &'a FxHashMap<ty::Region<'tcx>, RegionVid>,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for RegionIndexCollector<'_, 'tcx> {
    fn visit_region(&mut self, r: ty::Region<'tcx>) {
        let idx = self.walk_index;
        self.walk_index += 1;
        let maybe_vid = match r.kind() {
            ty::ReVar(vid) => Some(vid),
            _ => self.external_region_to_vid.get(&r).copied(),
        };
        if let Some(vid) = maybe_vid {
            debug_assert!(self.mappings.insert(idx as u32, vid.as_u32()).is_none());
            self.relevant_vids.insert(vid);
        }
        // Non-mapped regions (ReBound, ReErased, etc.) still consume
        // an index so that walk-order positions stay stable across
        // different region representations of the same type structure.
    }
}

struct InputSlotCollector<'a> {
    arg_ordinal: u32,
    offset_within_arg: u32,
    slots: &'a mut FxHashMap<u32, InputSlot>,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for InputSlotCollector<'_> {
    fn visit_region(&mut self, r: ty::Region<'tcx>) {
        let slot =
            InputSlot { arg_ordinal: self.arg_ordinal, offset_within_arg: self.offset_within_arg };
        self.offset_within_arg += 1;
        if let ty::ReEarlyParam(ep) = r.kind() {
            self.slots.entry(ep.index).or_insert(slot);
        }
    }
}

fn build_param_pos_to_input_slot<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
) -> FxHashMap<u32, InputSlot> {
    let mut slots = FxHashMap::default();
    let identity_args = ty::GenericArgs::identity_for_item(tcx, def_id);
    for (arg_ordinal, arg) in identity_args.iter().enumerate() {
        let mut collector = InputSlotCollector {
            arg_ordinal: arg_ordinal as u32,
            offset_within_arg: 0,
            slots: &mut slots,
        };
        arg.visit_with(&mut collector);
    }
    slots
}

fn input_slot_for_param_pos(
    param_pos: u32,
    param_pos_to_slot: &FxHashMap<u32, InputSlot>,
) -> InputSlot {
    param_pos_to_slot.get(&param_pos).copied().unwrap_or_else(|| {
        // Some region kinds (for example named late params) do not expose a
        // finer structural slot through the identity-arg walk. Fall back to
        // the param position so provenance still distinguishes input-sourced
        // regions from LocalOnly ones.
        InputSlot { arg_ordinal: param_pos, offset_within_arg: 0 }
    })
}

pub(crate) fn compute_region_summary<'tcx>(
    regioncx: &RegionInferenceContext<'tcx>,
    body: &Body<'tcx>,
    tcx: TyCtxt<'tcx>,
) -> BorrowckRegionSummary {
    let external_region_to_vid: FxHashMap<ty::Region<'tcx>, RegionVid> = regioncx
        .definitions
        .iter_enumerated()
        .filter_map(|(vid, def)| def.external_name.map(|region| (region, vid)))
        .collect();

    // ── Phase 1: Extract call-site region mappings ─────────────────
    //
    // Walk every terminator. For each Call whose callee resolves to
    // an FnDef, walk the callee's generic args with a TypeVisitor to
    // collect every region in depth-first visitation order. Each
    // region encountered gets a walk-order index (0, 1, 2, ...),
    // regardless of region kind. Only ReVar regions (inference
    // variables) produce entries — but non-ReVar regions (e.g.
    // ReBound from higher-ranked types) still advance the index, so
    // that positions stay aligned with the consumer's walk over the
    // same type structure after monomorphization.

    let mut call_site_mappings = UnordMap::default();
    let mut relevant_vids: FxIndexSet<RegionVid> = FxIndexSet::default();

    for bb_data in body.basic_blocks.iter() {
        let (func, call_id) = match bb_data.terminator().kind {
            TerminatorKind::Call { ref func, call_id, .. }
            | TerminatorKind::TailCall { ref func, call_id, .. } => (func, call_id),
            _ => continue,
        };
        // Borrowck runs before inlining, so all call chains must be
        // single-element at this point.
        debug_assert_eq!(
            call_id.len(),
            1,
            "expected single-element call chain in pre-inlining MIR, got {:?}",
            call_id,
        );
        let &(_, local_id, _) = &call_id[0];
        let func_ty = func.ty(&body.local_decls, tcx);
        if let ty::FnDef(_, args) = func_ty.kind() {
            let mut region_mappings = UnordMap::default();
            let mut collector = RegionIndexCollector {
                walk_index: 0,
                mappings: &mut region_mappings,
                relevant_vids: &mut relevant_vids,
                external_region_to_vid: &external_region_to_vid,
            };
            for arg in args.iter() {
                arg.visit_with(&mut collector);
            }
            if !region_mappings.is_empty() {
                call_site_mappings
                    .insert(local_id, CallSiteRegionMapping { call_id: local_id, region_mappings });
            }
        }
    }

    // Early exit: no call sites with lifetime params → empty summary.
    if call_site_mappings.is_empty() {
        return BorrowckRegionSummary::default();
    }

    // ── Phase 2: Project the SCC DAG ──────────────────────────────
    //
    // The full RegionInferenceContext partitions all RegionVids into
    // SCCs and stores a DAG over them. We project onto the relevant
    // set:
    //
    // - Each original SCC containing ≥1 relevant vid gets a
    //   projected SCC index. Relevant vids in the same original SCC
    //   share a projected SCC (they are mutually outliving).
    //
    // - A projected edge A → B exists iff original SCC A can reach
    //   original SCC B, possibly through intermediate SCCs that
    //   contain no relevant vids. This preserves transitive
    //   reachability while eliding invisible intermediaries.

    let constraint_sccs = regioncx.constraint_sccs();

    // Include universal region vids in the relevant set so that
    // inference variables and universal regions in the same SCC share
    // a projected SCC index. Without this, consumers cannot resolve
    // inference vids (from call-site generic args) to their
    // corresponding universal region param positions.
    for (vid, def) in regioncx.definitions.iter_enumerated() {
        if def.external_name.is_some() {
            relevant_vids.insert(vid);
        }
    }

    // Assign projected SCC indices.
    let mut orig_to_proj: FxIndexMap<ConstraintSccIndex, u32> = FxIndexMap::default();
    let mut next_proj: u32 = 0;
    let mut scc_of = UnordMap::default();

    for &vid in &relevant_vids {
        let orig_scc = constraint_sccs.scc(vid);
        let proj_idx = *orig_to_proj.entry(orig_scc).or_insert_with(|| {
            let idx = next_proj;
            next_proj += 1;
            idx
        });
        scc_of.insert(vid.as_u32(), proj_idx);
    }

    let num_proj = next_proj as usize;

    // DFS from each projected SCC through the original DAG.
    //
    // When we reach another projected SCC, record a projected edge
    // and stop — that SCC's own DFS handles onward reachability.
    // When we reach a non-projected SCC, continue through it (it is
    // an invisible intermediate).
    let mut scc_successors: Vec<Vec<u32>> = vec![Vec::new(); num_proj];

    for (&orig_scc, &proj_idx) in &orig_to_proj {
        let mut visited: FxHashSet<ConstraintSccIndex> = FxHashSet::default();
        visited.insert(orig_scc);
        let mut stack: Vec<ConstraintSccIndex> = constraint_sccs.successors(orig_scc).to_vec();

        while let Some(scc) = stack.pop() {
            if !visited.insert(scc) {
                continue;
            }
            if let Some(&target_proj) = orig_to_proj.get(&scc) {
                // Reached another projected SCC — record edge, stop.
                scc_successors[proj_idx as usize].push(target_proj);
            } else {
                // Non-projected SCC — continue through it.
                for &succ in constraint_sccs.successors(scc) {
                    stack.push(succ);
                }
            }
        }

        scc_successors[proj_idx as usize].sort();
        scc_successors[proj_idx as usize].dedup();
    }

    let outlives_graph = ProjectedOutlivesGraph { scc_of, scc_successors };

    // ── Phase 3: Universal region identity → param positions ─────
    //
    // Map every universal RegionVid to a param position:
    //   ReEarlyParam → ep.index
    //   ReLateParam (Named) → vid itself (identity)
    //   ReStatic → STATIC_PARAM_POS sentinel
    //
    // This covers all universal regions, not just those in the
    // relevant set — consumers may need to identify a vid as
    // 'static even if it did not appear in any call-site generic
    // args.

    let vid_to_param_pos: Vec<(u32, u32)> = regioncx
        .definitions
        .iter_enumerated()
        .filter_map(|(vid, def)| {
            let region = def.external_name?;
            let param_pos = match region.kind() {
                ty::ReEarlyParam(ep) => ep.index,
                ty::ReLateParam(lp) => match lp.kind {
                    ty::LateParamRegionKind::Named(_) => vid.as_u32(),
                    _ => return None,
                },
                ty::ReStatic => STATIC_PARAM_POS,
                _ => return None,
            };
            Some((vid.as_u32(), param_pos))
        })
        .collect();

    // ── Phase 4: Resolve call-site inference vids to params ────────
    //
    // For each call-site inference vid, check if it shares an SCC with
    // a non-static universal region in the FULL (unprojected) SCC graph.
    // SCC equality captures invariant subtyping (dyn types are invariant
    // in their lifetime parameters), which creates mutual constraints
    // between the call's inference vars and the corresponding universal
    // regions. This is more precise than reachability or raw-constraint
    // approaches, which incorrectly map vids that are merely outlived
    // by a universal region.

    // Build universal SCC → param_pos mapping.
    let mut univ_scc_to_param: FxIndexMap<ConstraintSccIndex, u32> = FxIndexMap::default();
    for &(vid, param_pos) in &vid_to_param_pos {
        if param_pos == STATIC_PARAM_POS {
            continue;
        }
        let scc = constraint_sccs.scc(RegionVid::from_u32(vid));
        univ_scc_to_param.insert(scc, param_pos);
    }

    let mut vid_to_resolved_param_map: FxHashMap<u32, u32> = FxHashMap::default();
    for &vid in &relevant_vids {
        if regioncx.definitions[vid].external_name.is_some() {
            continue;
        }
        let vid_scc = constraint_sccs.scc(vid);
        if let Some(&param_pos) = univ_scc_to_param.get(&vid_scc) {
            vid_to_resolved_param_map.insert(vid.as_u32(), param_pos);
        }
    }

    // ── Phase 4b: Compute "bounded-by" resolution ────────────────────
    //
    // For inference vids that are NOT in a universal SCC (due to
    // covariant dyn-type constraints generating only forward edges),
    // check if the vid's SCC is reachable from exactly one non-static
    // universal SCC. Post-borrowck, the concrete lifetime through the
    // unsizing edge IS that universal — the covariance permission
    // doesn't change the runtime value.

    // Precompute transitive reachability over the SCC DAG as two
    // bit matrices:
    //   `forward[scc]`  = descendants of `scc` ∪ {scc}
    //   `reverse[scc]`  = ancestors of `scc`  ∪ {scc}
    // Both are built with a single fold. `all_sccs()` yields post-order
    // (successors before predecessors), so `forward` reads from
    // already-complete rows when processing an SCC. `reverse` needs the
    // opposite order — iterate `.rev()` so every scc's ancestors are
    // finalized before being propagated into its successors.
    let num_sccs = constraint_sccs.num_sccs();
    let mut forward: SparseBitMatrix<ConstraintSccIndex, ConstraintSccIndex> =
        SparseBitMatrix::new(num_sccs);
    for scc in constraint_sccs.all_sccs() {
        forward.insert(scc, scc);
        for &succ in constraint_sccs.successors(scc) {
            // Edge scc -> succ: scc's row absorbs succ's (completed) row.
            forward.union_rows(succ, scc);
        }
    }
    let mut reverse: SparseBitMatrix<ConstraintSccIndex, ConstraintSccIndex> =
        SparseBitMatrix::new(num_sccs);
    // `all_sccs()` only exposes `impl Iterator`, not `DoubleEndedIterator`,
    // so reverse-post-order is expressed directly over the index range.
    for scc in (0..num_sccs).rev().map(ConstraintSccIndex::new) {
        reverse.insert(scc, scc);
        for &succ in constraint_sccs.successors(scc) {
            // Edge scc -> succ: push scc's (completed) ancestors into succ.
            reverse.union_rows(scc, succ);
        }
    }

    // Record which universal-SCC param positions transitively reach
    // each descendant SCC. A universal is not listed among its own
    // predecessors.
    let mut scc_to_univ_predecessors: FxHashMap<ConstraintSccIndex, Vec<u32>> =
        FxHashMap::default();
    for (&univ_scc, &param_pos) in &univ_scc_to_param {
        for reached in forward.iter(univ_scc) {
            if reached == univ_scc {
                continue;
            }
            scc_to_univ_predecessors.entry(reached).or_default().push(param_pos);
        }
    }

    // Compute the full set of SCCs reachable from 'static, used to
    // identify non-static predecessors.
    let static_vid = RegionVid::from_u32(0);
    let static_scc = constraint_sccs.scc(static_vid);
    let mut reachable_from_static = FxHashSet::default();
    {
        let mut stack = vec![static_scc];
        reachable_from_static.insert(static_scc);
        while let Some(scc) = stack.pop() {
            for &succ in constraint_sccs.successors(scc) {
                if reachable_from_static.insert(succ) {
                    stack.push(succ);
                }
            }
        }
    }

    let vid_to_param_pos_map: FxHashMap<u32, u32> = vid_to_param_pos.iter().copied().collect();
    let input_slot_by_param_pos = build_param_pos_to_input_slot(tcx, body.source.def_id());

    // Scratch reused across vids for the domination check below.
    let mut scratch: DenseBitSet<ConstraintSccIndex> = DenseBitSet::new_empty(num_sccs);
    let mut vid_provenance = UnordMap::default();
    for &vid in &relevant_vids {
        let vid_u32 = vid.as_u32();
        let provenance = if let Some(&param_pos) = vid_to_param_pos_map.get(&vid_u32) {
            if param_pos == STATIC_PARAM_POS {
                VidProvenance::Static
            } else {
                VidProvenance::Input(input_slot_for_param_pos(param_pos, &input_slot_by_param_pos))
            }
        } else if let Some(&param_pos) = vid_to_resolved_param_map.get(&vid_u32) {
            VidProvenance::Input(input_slot_for_param_pos(param_pos, &input_slot_by_param_pos))
        } else {
            // Check bounded-by: does exactly one non-static universal
            // SCC U reach this vid's SCC S, and does U dominate S
            // (every predecessor of S is also reachable from U)?
            //
            // The domination check ensures that no local-only lifetime
            // feeds into S from a path that doesn't go through U.
            // Without it, an SCC merged through trait bounds (e.g.,
            // Sub4<'a>: Super4<'a,'a>) would be classified as bounded
            // by 'a even when some vids represent a local lifetime.
            let vid_scc = constraint_sccs.scc(vid);
            match scc_to_univ_predecessors.get(&vid_scc) {
                Some(preds) if preds.len() == 1 => {
                    // Find the universal SCC for this param_pos.
                    let univ_scc = univ_scc_to_param
                        .iter()
                        .find(|&(_, &pp)| pp == preds[0])
                        .map(|(&scc, _)| scc);
                    // Domination: every non-static, non-U ancestor of
                    // `vid_scc` must also be reachable from `u_scc`.
                    // Expressed as: ancestors(vid_scc) \ {static, u_scc}
                    //               ⊆ descendants(u_scc) ∪ {u_scc}.
                    // `vid_scc` itself is in `ancestors(vid_scc)` but also
                    // in `forward.row(u_scc)` — since we only reach this
                    // arm when `u_scc` reaches `vid_scc` — so it cancels.
                    let dominated = univ_scc.is_some_and(|u_scc| {
                        match (reverse.row(vid_scc), forward.row(u_scc)) {
                            (Some(ancestors), Some(fwd_u)) => {
                                scratch.clone_from(ancestors);
                                scratch.remove(static_scc);
                                scratch.remove(u_scc);
                                scratch.subtract(fwd_u);
                                scratch.is_empty()
                            }
                            // `vid_scc` has no ancestors — trivially dominated.
                            (None, _) => true,
                            // Defensive: `forward` self-insertion guarantees
                            // `u_scc` has a row, so this arm is unreachable.
                            (Some(_), None) => false,
                        }
                    });
                    if dominated {
                        VidProvenance::BoundedByUniversal(input_slot_for_param_pos(
                            preds[0],
                            &input_slot_by_param_pos,
                        ))
                    } else {
                        VidProvenance::LocalOnly
                    }
                }
                _ => VidProvenance::LocalOnly,
            }
        };
        vid_provenance.insert(vid_u32, provenance);
    }

    BorrowckRegionSummary { call_site_mappings, outlives_graph, vid_provenance, vid_to_param_pos }
}
