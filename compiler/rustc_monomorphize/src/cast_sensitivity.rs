//! SCC-based batch computation of cast-relevant lifetimes and related
//! query providers. Composes per-Instance direct sensitivity with callee
//! sensitivity over the call graph; the SCC pass is order-independent so
//! back-edges propagate correctly.

use std::collections::VecDeque;

use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_data_structures::graph::scc::Sccs;
use rustc_data_structures::graph::vec_graph::VecGraph;
use rustc_data_structures::sync::{Lock, par_map};
use rustc_data_structures::unord::{ExtendUnord, UnordMap, UnordSet};
use rustc_hir::def_id::DefId;
use rustc_index::bit_set::{BitMatrix, DenseBitSet};
use rustc_middle::bug;
use rustc_middle::mir::{self, BorrowckRegionSummary, InputSlot, VidProvenance};
use rustc_middle::mono::{
    CastRelevantLifetimes, CollectionMode, LifetimeBVToParamMapping, MonoItem,
};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, Instance, List, TyCtxt};
use rustc_span::{DUMMY_SP, sym};
use smallvec::SmallVec;

use crate::erasure_safe::{region_slots_of_arg, region_slots_of_args};

// ── Types ─────────────────────────────────────────────────────────────────

/// Per-Instance sensitivity metadata, keyed on base Instance
/// (no Outlives entries). Populated by the SCC batch computation and
/// consumed when augmenting sensitive subgraphs.
pub(crate) struct InstanceSensitivity<'tcx> {
    /// Transitive sensitivity: composed from direct + callee sensitivity.
    /// `None` → this Instance is not sensitive.
    pub(crate) sensitivity: Option<CastRelevantLifetimes<'tcx>>,
    /// The base (un-augmented) sensitive callees at each call site.
    /// Stored so augmentation can re-run with a different `CallerOutlivesEnv`.
    /// Empty for directly sensitive functions (they are the leaves).
    pub(crate) sensitive_call_sites:
        &'tcx [(&'tcx List<(DefId, u32, ty::GenericArgsRef<'tcx>)>, Instance<'tcx>)],
    /// For ground-level callers: pre-computed augmented callee Instances.
    /// Empty for generic callers, whose augmented callees depend on
    /// Outlives entries computed later when a concrete caller is known.
    pub(crate) augmented_callees:
        &'tcx [(&'tcx List<(DefId, u32, ty::GenericArgsRef<'tcx>)>, Instance<'tcx>)],
}

/// Outlives oracle for a caller. Pre-computes a Floyd-Warshall
/// reachability matrix (via the `outlives_reachability` query) at
/// construction time for O(1) `outlives()` lookups. Two cases:
/// - Augmented callers: built from Instance Outlives entries.
/// - Ground-level callers: keys are walk positions from the origin call
///   site, translated through the call-site region mapping before lookup.
#[derive(Debug)]
pub(crate) struct CallerOutlivesEnv<'tcx> {
    reach: &'tcx BitMatrix<usize, usize>,
    dim: usize,
    /// Maps caller-space keys (walk positions or binder indices) to
    /// matrix indices. `None` when the caller-space keys ARE matrix
    /// indices (the `FromOutlivesEntries` path).
    key_to_idx: Option<FxHashMap<usize, usize>>,
}

impl<'tcx> CallerOutlivesEnv<'tcx> {
    /// Build from a pre-computed reachability matrix where caller-space
    /// keys are direct matrix indices (no remapping).
    pub(crate) fn from_raw(reach: &'tcx BitMatrix<usize, usize>, dim: usize) -> Self {
        CallerOutlivesEnv { reach, dim, key_to_idx: None }
    }

    pub(crate) fn from_outlives_entries(tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>) -> Self {
        // Single pass: intern the outlives args directly into the arena
        // while tracking the largest live index. `max_idx` stays 0 when
        // the iterator is empty; `interned.is_empty()` covers the empty
        // case below.
        let mut max_idx = 0;
        let interned = tcx.arena.alloc_from_iter(instance.outlives_indices_iter().map(|(l, s)| {
            for v in [l, s] {
                if v != usize::MAX && v > max_idx {
                    max_idx = v;
                }
            }
            tcx.mk_outlives_arg(l, s).into()
        }));
        let dim = if interned.is_empty() { 1 } else { max_idx + 2 };
        let reach = tcx.outlives_reachability((interned, dim));
        CallerOutlivesEnv { reach, dim, key_to_idx: None }
    }

    /// Build an env where keys are walk positions from a specific call site.
    /// Converts the SCC graph edges into outlives entries and pre-computes
    /// the reachability matrix.
    pub(crate) fn from_region_summary_walk_pos(
        tcx: TyCtxt<'tcx>,
        summary: &'tcx BorrowckRegionSummary,
        call_site_mapping: &mir::CallSiteRegionMapping,
    ) -> Self {
        let num_sccs = summary.outlives_graph.scc_successors.len();
        let dim = num_sccs + 1; // +1 for 'static slot
        let static_idx = dim - 1;

        let interned = tcx.arena.alloc_from_iter(
            summary.outlives_graph.scc_successors.iter().enumerate().flat_map(
                |(from_scc, successors)| {
                    successors
                        .iter()
                        .map(move |&to_scc| tcx.mk_outlives_arg(from_scc, to_scc as usize).into())
                },
            ),
        );
        let reach = tcx.outlives_reachability((interned, dim));

        let key_to_idx: FxHashMap<usize, usize> = call_site_mapping
            .region_mappings
            .items()
            .map(|(&walk_pos, &vid)| {
                let scc = summary.outlives_graph.scc_of_vid(vid).unwrap_or_else(|| {
                    bug!("missing SCC mapping for region vid {vid} in {:?}", summary)
                });
                (walk_pos as usize, scc as usize)
            })
            .into_sorted_stable_ord()
            .into_iter()
            .collect();

        // Check if any walk position maps to the 'static SCC. The
        // projected graph doesn't have an explicit 'static SCC — it's
        // implicit. Walk positions mapping to vids whose SCC reaches
        // everything are already handled by the reachability matrix.
        // The static_idx slot is added by outlives_reachability.
        let _ = static_idx;

        CallerOutlivesEnv { reach, dim, key_to_idx: Some(key_to_idx) }
    }

    /// Matrix index reserved for `'static` (always `dim - 1`).
    pub(crate) fn static_idx(&self) -> usize {
        self.dim - 1
    }

    /// Resolve a caller-space key to a matrix index, or `None` if the key
    /// is absent from the environment. `usize::MAX` always maps to
    /// `static_idx`.
    pub(crate) fn resolve(&self, key: usize) -> Option<usize> {
        if key == usize::MAX {
            return Some(self.static_idx());
        }
        match &self.key_to_idx {
            None => (key < self.dim).then_some(key),
            Some(map) => map.get(&key).copied(),
        }
    }

    /// Iterate all matrix indices that `idx` reaches (i.e. all `shorter`
    /// such that `idx` outlives `shorter`).
    pub(crate) fn reach_row(&self, idx: usize) -> impl Iterator<Item = usize> + '_ {
        self.reach.iter(idx)
    }

    /// Query whether `longer` outlives `shorter` in the caller's environment.
    pub(crate) fn outlives(&self, longer: usize, shorter: usize) -> bool {
        if longer == shorter {
            return true;
        }
        let Some(l) = self.resolve(longer) else { return false };
        let Some(s) = self.resolve(shorter) else { return false };
        self.reach.contains(l, s)
    }
}

#[derive(Clone)]
struct CallEdge<'tcx> {
    call_id: &'tcx List<(DefId, u32, ty::GenericArgsRef<'tcx>)>,
    callee: Instance<'tcx>,
}

#[derive(Clone, Copy)]
struct ConcretizedChainEdge<'tcx> {
    body_args: ty::GenericArgsRef<'tcx>,
    concrete_edge_args: ty::GenericArgsRef<'tcx>,
}

// ── Composition helpers ───────────────────────────────────────────────────

struct InputDecomposition {
    to_walk_pos: FxHashMap<InputSlot, usize>,
    #[allow(dead_code)]
    from_walk_pos: FxHashMap<usize, InputSlot>,
}

fn build_input_decomposition<'tcx>(concrete_args: ty::GenericArgsRef<'tcx>) -> InputDecomposition {
    let mut to_walk_pos = FxHashMap::default();
    let mut from_walk_pos = FxHashMap::default();
    let mut walk_pos = 0usize;

    for (arg_ordinal, arg) in concrete_args.iter().enumerate() {
        let slots = region_slots_of_arg(arg);
        for offset in 0..slots {
            let wp = walk_pos + offset;
            let slot =
                InputSlot { arg_ordinal: arg_ordinal as u32, offset_within_arg: offset as u32 };
            to_walk_pos.insert(slot, wp);
            from_walk_pos.insert(wp, slot);
        }
        walk_pos += slots;
    }

    InputDecomposition { to_walk_pos, from_walk_pos }
}

fn resolve_vid_provenance(vid: u32, summary: &BorrowckRegionSummary) -> VidProvenance {
    summary
        .vid_provenance
        .get(&vid)
        .copied()
        .unwrap_or_else(|| bug!("relevant vid {vid} missing provenance in {:?}", summary))
}

fn instance_from_edge_args_or_bug<'tcx>(
    tcx: TyCtxt<'tcx>,
    next_body_def_id: DefId,
    concrete_edge_args: ty::GenericArgsRef<'tcx>,
) -> Instance<'tcx> {
    let instance = Instance::expect_resolve(
        tcx,
        ty::TypingEnv::fully_monomorphized(),
        next_body_def_id,
        concrete_edge_args,
        DUMMY_SP,
    );

    if instance.def_id() != next_body_def_id {
        bug!(
            "call_id chain/body mismatch: resolved {:?} for next body {:?} with args {:?}",
            instance,
            next_body_def_id,
            concrete_edge_args,
        );
    }

    match instance.def {
        ty::InstanceKind::Virtual(..) | ty::InstanceKind::Intrinsic(..) => {
            bug!(
                "next call_id body {:?} resolved to non-MIR instance {:?}",
                next_body_def_id,
                instance,
            );
        }
        _ => instance,
    }
}

fn concretize_chain_args<'tcx>(
    tcx: TyCtxt<'tcx>,
    caller: Instance<'tcx>,
    call_id: &[(DefId, u32, ty::GenericArgsRef<'tcx>)],
) -> Vec<ConcretizedChainEdge<'tcx>> {
    let mut result = Vec::with_capacity(call_id.len());
    let mut current_instance = caller;

    for (i, &(body_def_id, _local_id, edge_args_template)) in call_id.iter().enumerate() {
        if current_instance.def_id() != body_def_id {
            bug!(
                "call_id chain out of sync: expected body {:?}, got instance {:?}",
                body_def_id,
                current_instance,
            );
        }

        let concrete_args = current_instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            ty::TypingEnv::fully_monomorphized(),
            ty::EarlyBinder::bind(edge_args_template),
        );
        result.push(ConcretizedChainEdge {
            body_args: current_instance.args,
            concrete_edge_args: concrete_args,
        });

        if let Some(&(next_body_def_id, _next_local_id, _)) = call_id.get(i + 1) {
            current_instance = instance_from_edge_args_or_bug(tcx, next_body_def_id, concrete_args);
        }
    }

    result
}

fn build_template_input_slot_map<'tcx>(
    body_args: ty::GenericArgsRef<'tcx>,
    edge_args_template: ty::GenericArgsRef<'tcx>,
    concrete_edge_args: ty::GenericArgsRef<'tcx>,
) -> FxHashMap<usize, InputSlot> {
    let mut walk_pos_to_slot = FxHashMap::default();
    let mut walk_pos = 0usize;

    for (template_arg, concrete_arg) in edge_args_template.iter().zip(concrete_edge_args.iter()) {
        // Determine which of the caller's generic params this template
        // arg forwards. Non-forwarding args (concrete types, literal
        // consts, etc.) are skipped — their walk positions get no
        // InputSlot mapping, which downstream treats as unresolvable.
        let source_arg_ordinal = match template_arg.kind() {
            ty::GenericArgKind::Type(ty) => match ty.kind() {
                ty::Param(param_ty) => Some(param_ty.index as usize),
                _ => None,
            },
            ty::GenericArgKind::Lifetime(region) => match region.kind() {
                ty::ReEarlyParam(ep) => Some(ep.index as usize),
                _ => None,
            },
            ty::GenericArgKind::Const(ct) => match ct.kind() {
                ty::ConstKind::Param(param_ct) => Some(param_ct.index as usize),
                _ => None,
            },
            ty::GenericArgKind::Outlives(_) => None,
        };

        let slots = region_slots_of_arg(concrete_arg);
        if let Some(source_ordinal) = source_arg_ordinal {
            debug_assert!(
                body_args.get(source_ordinal).is_some(),
                "template arg referenced missing body arg {} in {:?}",
                source_ordinal,
                body_args
            );
            for offset in 0..slots {
                walk_pos_to_slot.insert(
                    walk_pos + offset,
                    InputSlot {
                        arg_ordinal: source_ordinal as u32,
                        offset_within_arg: offset as u32,
                    },
                );
            }
        }
        walk_pos += slots;
    }

    walk_pos_to_slot
}

/// Compose all walk-order positions through a call_id chain in one pass.
///
/// The output stays in the origin call site's walk-position space. This
/// lets callers compare the transported positions directly against the
/// caller's own outlives environment without collapsing them to SCC ids.
pub(crate) fn compose_all_through_chain<'tcx>(
    tcx: TyCtxt<'tcx>,
    caller: Instance<'tcx>,
    call_id: &[(DefId, u32, ty::GenericArgsRef<'tcx>)],
    n_positions: usize,
) -> Vec<Option<usize>> {
    let mut positions: Vec<Option<usize>> = (0..n_positions).map(Some).collect();
    if call_id.is_empty() || n_positions == 0 {
        return positions;
    }

    let dump_filter = tcx.sess.opts.unstable_opts.dump_trait_cast_chain_composition.as_deref();
    let dump = match dump_filter {
        Some(f) if f == "all" => true,
        Some(f) => with_no_trimmed_paths!(caller.to_string()).contains(f),
        None => false,
    };

    if dump {
        let caller_name = with_no_trimmed_paths!(caller.to_string());
        eprintln!(
            "=== Chain Composition: {caller_name} ({} link(s), max_walk_pos={n_positions}) ===",
            call_id.len(),
        );
    }

    let links: Vec<_> = call_id.iter().rev().copied().collect();
    let concrete_args_per_edge = concretize_chain_args(tcx, caller, call_id);

    for (i, &(body_def_id, local_id, _)) in links.iter().enumerate() {
        let edge_index = call_id.len() - 1 - i;
        let edge_info = concrete_args_per_edge[edge_index];
        let concrete_edge_args = edge_info.concrete_edge_args;
        let summary = tcx.borrowck_region_summary(body_def_id);
        let is_outermost = i + 1 == links.len();
        let outer_decomp = if is_outermost {
            None
        } else {
            let outer_edge_index = call_id.len() - (i + 2);
            Some(build_input_decomposition(
                concrete_args_per_edge[outer_edge_index].concrete_edge_args,
            ))
        };

        if dump {
            let body_path = with_no_trimmed_paths!(tcx.def_path_str(body_def_id));
            let template_args = with_no_trimmed_paths!(format!("{:?}", call_id[edge_index].2));
            let body_args_s = with_no_trimmed_paths!(format!("{:?}", edge_info.body_args));
            let concrete_args_s = with_no_trimmed_paths!(format!("{:?}", concrete_edge_args));
            eprintln!("  Link {i}: body={body_path} local_id={local_id}");
            eprintln!("    body_args (before this link's edge): {body_args_s}");
            eprintln!("    edge args (template): {template_args}");
            eprintln!("    edge args (concretized): {concrete_args_s}");

            let template_map = build_template_input_slot_map(
                edge_info.body_args,
                call_id[edge_index].2,
                concrete_edge_args,
            );
            if template_map.is_empty() {
                eprintln!("    Template walk_pos -> InputSlot: (empty)");
            } else {
                eprintln!("    Template walk_pos -> InputSlot:");
                #[allow(rustc::potential_query_instability)]
                // Collecting map entries into a Vec that we subsequently sort by
                // the usize walk_pos key for deterministic diagnostic output.
                let mut entries: Vec<(usize, InputSlot)> =
                    template_map.iter().map(|(&k, &v)| (k, v)).collect();
                entries.sort_by_key(|&(k, _)| k);
                for (wp, slot) in entries {
                    eprintln!(
                        "      walk_pos={wp} -> InputSlot {{ arg_ordinal={}, offset_within_arg={} }}",
                        slot.arg_ordinal, slot.offset_within_arg,
                    );
                }
            }
        }

        let Some(mapping) = summary.call_site_mappings.get(&local_id) else {
            if dump {
                eprintln!("    Region-summary resolutions: (no call_site_mapping for local_id)");
            }
            let edge_slots: usize = region_slots_of_args(concrete_edge_args);
            if edge_slots == 0 {
                if dump {
                    eprintln!(
                        "    edge has 0 region slots after monomorphization; dropping all positions"
                    );
                }
                positions.fill(None);
                continue;
            }

            // No call-site mapping but regions exist after monomorphization
            // (e.g. U = dyn Trait<'lt>). Trace through the template args.
            let local_walk_pos_to_slot = build_template_input_slot_map(
                edge_info.body_args,
                call_id[edge_index].2,
                concrete_edge_args,
            );

            // At the outermost link there is no outer_decomp — use the
            // caller's own args as the target coordinate space.
            let caller_decomp;
            let target_decomp = if is_outermost {
                caller_decomp = build_input_decomposition(edge_info.body_args);
                &caller_decomp
            } else {
                outer_decomp.as_ref().expect("non-outermost link must have an outer decomposition")
            };

            for pos in positions.iter_mut() {
                let Some(wp) = *pos else {
                    continue;
                };
                let Some(slot) = local_walk_pos_to_slot.get(&wp).copied() else {
                    *pos = None;
                    continue;
                };
                let Some(&outer_wp) = target_decomp.to_walk_pos.get(&slot) else {
                    *pos = None;
                    continue;
                };
                *pos = Some(outer_wp);
            }
            continue;
        };

        if dump {
            let entries: Vec<(u32, u32)> = mapping
                .region_mappings
                .items()
                .map(|(&walk_pos, &vid)| (walk_pos, vid))
                .into_sorted_stable_ord();
            if entries.is_empty() {
                eprintln!("    Region-summary resolutions: (call_site_mapping is empty)");
            } else {
                eprintln!("    Region-summary resolutions:");
                for (wp, vid) in entries {
                    let prov = resolve_vid_provenance(vid, &summary);
                    eprintln!("      walk_pos={wp} -> vid={vid} -> {prov:?}");
                }
            }
        }

        for pos in positions.iter_mut() {
            let Some(wp) = *pos else {
                continue;
            };

            let Some(vid) = mapping.vid_for_walk_pos(wp as u32) else {
                *pos = None;
                continue;
            };

            if is_outermost {
                if matches!(resolve_vid_provenance(vid, &summary), VidProvenance::Static) {
                    *pos = None;
                } else {
                    *pos = Some(wp);
                }
                continue;
            }

            match resolve_vid_provenance(vid, &summary) {
                VidProvenance::Static | VidProvenance::LocalOnly => {
                    *pos = None;
                }
                VidProvenance::Input(slot) | VidProvenance::BoundedByUniversal(slot) => {
                    let Some(&outer_wp) = outer_decomp
                        .as_ref()
                        .expect("non-outermost link must have an outer decomposition")
                        .to_walk_pos
                        .get(&slot)
                    else {
                        *pos = None;
                        continue;
                    };
                    *pos = Some(outer_wp);
                }
            }
        }
    }

    if dump {
        eprintln!("  Final mapping (callee walk_pos -> origin walk_pos):");
        for (i, entry) in positions.iter().enumerate() {
            match entry {
                Some(n) if *n == usize::MAX => eprintln!("    [{i}] -> 'static"),
                Some(n) => eprintln!("    [{i}] -> {n}"),
                None => eprintln!("    [{i}] -> (none)"),
            }
        }
        eprintln!();
    }

    positions
}

fn caller_env_for_call_id<'tcx>(
    tcx: TyCtxt<'tcx>,
    caller: Instance<'tcx>,
    call_id: &'tcx List<(DefId, u32, ty::GenericArgsRef<'tcx>)>,
) -> CallerOutlivesEnv<'tcx> {
    if call_id.is_empty() {
        bug!("empty call_id chain for caller {:?}", caller);
    }
    if caller.has_outlives_entries() {
        return CallerOutlivesEnv::from_outlives_entries(tcx, &caller);
    }

    let origin_def_id = call_id[0].0;
    debug_assert_eq!(origin_def_id, caller.def_id());
    let origin_local_id = call_id[0].1;
    let summary = tcx.borrowck_region_summary(origin_def_id);
    let Some(mapping) = summary.call_site_mappings.get(&origin_local_id) else {
        // No call-site mapping: the origin function is generic and its
        // intrinsic args are type params that only acquire regions after
        // monomorphization. Return an empty env — outlives evidence will
        // come from the augmented Instance's Outlives entries when the
        // sensitivity system processes augmented callees.
        let empty: &'tcx [ty::GenericArg<'tcx>] = &[];
        return CallerOutlivesEnv::from_raw(tcx.outlives_reachability((empty, 1)), 1);
    };
    CallerOutlivesEnv::from_region_summary_walk_pos(tcx, summary, mapping)
}

// ── CastRelevantLifetimes helpers ─────────────────────────────────────────

fn input_identity_sensitivity_for_call_site<'tcx>(
    tcx: TyCtxt<'tcx>,
    summary: &'tcx BorrowckRegionSummary,
    mapping: &mir::CallSiteRegionMapping,
) -> Option<CastRelevantLifetimes<'tcx>> {
    // Allocate `bv_to_param` lazily: if no walk-pos survives the provenance
    // filter, the whole function returns None and we never pay for the vec.
    let mut bv_to_param: Option<Vec<Option<usize>>> = None;

    for (walk_pos, region_vid) in mapping
        .region_mappings
        .items()
        .map(|(&walk_pos, &region_vid)| (walk_pos, region_vid))
        .into_sorted_stable_ord()
    {
        if matches!(
            summary.vid_provenance.get(&region_vid),
            Some(
                VidProvenance::Input(_)
                    | VidProvenance::BoundedByUniversal(_)
                    | VidProvenance::LocalOnly
            )
        ) {
            let slots = bv_to_param.get_or_insert_with(|| {
                let max_walk_pos =
                    mapping.region_mappings.items().map(|(&wp, _)| wp as usize).max().unwrap_or(0);
                vec![None; max_walk_pos + 1]
            });
            slots[walk_pos as usize] = Some(walk_pos as usize);
        }
    }

    let bv_to_param = bv_to_param?;
    let list = tcx.mk_lifetime_bv_to_param_mapping_from_iter(bv_to_param.into_iter());
    let mappings = [LifetimeBVToParamMapping(list)];
    Some(CastRelevantLifetimes::from_direct_mappings(&mappings))
}

// ── augment_callee ────────────────────────────────────────────────────────

/// Compute the augmented callee Instance given the caller's outlives
/// environment and a pre-composed mapping from callee walk-order positions
/// to caller param positions.
pub(crate) fn augment_callee<'tcx>(
    tcx: TyCtxt<'tcx>,
    caller_instance: Instance<'tcx>,
    callee_instance: Instance<'tcx>,
    callee_sensitivity: &CastRelevantLifetimes<'tcx>,
    caller_env: &CallerOutlivesEnv<'tcx>,
    composed_mapping: Option<&[Option<usize>]>,
) -> Instance<'tcx> {
    // Determinism is established by `into_sorted_stable_ord` below, so we
    // can traverse the unord mappings directly without an up-front sort.
    let mut nodes: Vec<(usize, usize)> = callee_sensitivity
        .mappings
        .items()
        .flat_map(|mapping| {
            mapping.0.iter().enumerate().filter_map(|(bv_idx, bv)| {
                let callee_pos = bv?;
                // When composed_mapping is None, use identity (pass
                // CRL values through directly as caller param keys).
                let caller_param_pos = match composed_mapping {
                    Some(cm) => match cm.get(callee_pos) {
                        Some(&Some(pos)) => pos,
                        _ => return None,
                    },
                    None => callee_pos,
                };
                Some((bv_idx, caller_param_pos))
            })
        })
        .into_sorted_stable_ord();

    if nodes.is_empty() {
        let result = callee_instance.with_outlives(tcx, &[]);
        maybe_dump_augmentation(
            tcx,
            caller_instance,
            callee_instance,
            caller_env,
            composed_mapping,
            &nodes,
            &[],
            result,
        );
        return result;
    }

    // `into_sorted_stable_ord` above sorted lexicographically by
    // `(bv_idx, caller_param_pos)`, so same-`bv_idx` entries are already
    // adjacent and the dedup below drops duplicates per `bv_idx`.
    nodes.dedup_by_key(|n| n.0);

    // Resolve every node's caller-space key to a matrix index once, then
    // build the outlives pairs by iterating each row of the pre-computed
    // reachability matrix. This is O(N · dim) in the number of sensitive
    // binder variables and the matrix dimension (typically ≤ 10), rather
    // than the O(N²) pairwise `outlives` probes it replaces.
    let static_idx = caller_env.static_idx();
    let resolved: Vec<(usize, usize)> = nodes
        .iter()
        .filter_map(|&(bv, key)| caller_env.resolve(key).map(|idx| (bv, idx)))
        .collect();

    // Reverse index: matrix idx → bv(s) at that idx. `dim` is tiny, so this
    // map stays small; SmallVec inline-4 covers the typical aliasing case.
    let mut idx_to_bvs: FxHashMap<usize, SmallVec<[usize; 4]>> = FxHashMap::default();
    for &(bv, idx) in &resolved {
        idx_to_bvs.entry(idx).or_default().push(bv);
    }

    let mut outlives_pairs: Vec<(usize, usize)> = Vec::new();
    for &(bv_i, idx_i) in &resolved {
        for idx_j in caller_env.reach_row(idx_i) {
            // `'static` successor contributes the `(bv_i, usize::MAX)`
            // sentinel. Also fall through so any bv whose key was
            // `usize::MAX` (and thus resolved to `static_idx`) still
            // gets a pair emitted against it.
            if idx_j == static_idx {
                outlives_pairs.push((bv_i, usize::MAX));
            }
            // When multiple bvs alias onto the same caller-space key,
            // reflexivity (`reach.contains(idx, idx) == true`) still
            // relates them, so we do not skip `idx_j == idx_i`: the
            // `bv_i != bv_j` filter below rejects only true self-pairs.
            if let Some(bvs) = idx_to_bvs.get(&idx_j) {
                for &bv_j in bvs {
                    if bv_i != bv_j {
                        outlives_pairs.push((bv_i, bv_j));
                    }
                }
            }
        }
    }

    outlives_pairs.sort();
    outlives_pairs.dedup();

    let result = callee_instance.with_outlives(tcx, &outlives_pairs);
    maybe_dump_augmentation(
        tcx,
        caller_instance,
        callee_instance,
        caller_env,
        composed_mapping,
        &nodes,
        &outlives_pairs,
        result,
    );
    result
}

/// Emit the `-Z dump-trait-cast-augmentation` diagnostic block for a single
/// caller -> callee augmentation, when the flag is set and the filter matches
/// the caller's printed name. Pure instrumentation — does not affect the
/// augmentation result.
fn maybe_dump_augmentation<'tcx>(
    tcx: TyCtxt<'tcx>,
    caller_instance: Instance<'tcx>,
    callee_instance: Instance<'tcx>,
    caller_env: &CallerOutlivesEnv<'tcx>,
    composed_mapping: Option<&[Option<usize>]>,
    nodes: &[(usize, usize)],
    outlives_pairs: &[(usize, usize)],
    result: Instance<'tcx>,
) {
    let Some(ref filter) = tcx.sess.opts.unstable_opts.dump_trait_cast_augmentation else {
        return;
    };

    let caller_name = with_no_trimmed_paths!(caller_instance.to_string());
    if filter != "all" && !caller_name.contains(filter.as_str()) {
        return;
    }

    let callee_name = with_no_trimmed_paths!(callee_instance.to_string());
    let result_name = with_no_trimmed_paths!(result.to_string());

    eprintln!("=== Augmentation: {caller_name} ->");
    eprintln!("                  {callee_name} ===");

    eprintln!("  Caller outlives env:");
    {
        let dim = caller_env.dim;
        let static_idx = dim - 1;
        let has_remap = caller_env.key_to_idx.is_some();
        eprintln!("    CallerOutlivesEnv (dim={dim}, remapped={has_remap}):");
        if let Some(ref key_map) = caller_env.key_to_idx {
            #[allow(rustc::potential_query_instability)]
            let mut pairs: Vec<(usize, usize)> = key_map.iter().map(|(&k, &v)| (k, v)).collect();
            pairs.sort_by_key(|&(k, _)| k);
            for (walk_pos, idx) in pairs {
                eprintln!("      walk_pos={walk_pos} -> idx={idx}");
            }
        }
        for i in 0..dim {
            for j in 0..dim {
                if i != j && caller_env.reach.contains(i, j) {
                    let i_s = if i == static_idx { "'static".to_string() } else { i.to_string() };
                    let j_s = if j == static_idx { "'static".to_string() } else { j.to_string() };
                    eprintln!("      {i_s} outlives {j_s}");
                }
            }
        }
    }

    eprintln!("  Composed mapping:");
    match composed_mapping {
        None => eprintln!("    (identity)"),
        Some(cm) => {
            for (callee_walk_pos, entry) in cm.iter().enumerate() {
                match entry {
                    Some(pos) => {
                        eprintln!(
                            "    callee_walk_pos={callee_walk_pos} -> caller_param_pos={pos}"
                        );
                    }
                    None => {
                        eprintln!("    callee_walk_pos={callee_walk_pos} -> (none)");
                    }
                }
            }
        }
    }

    eprintln!("  BV nodes ({}):", nodes.len());
    for &(bv_idx, caller_param_pos) in nodes {
        let key = if caller_param_pos == usize::MAX {
            "'static".to_string()
        } else {
            caller_param_pos.to_string()
        };
        eprintln!("    bv{bv_idx} -> caller_param_pos={key}");
    }

    eprintln!("  Outlives pairs emitted ({}):", outlives_pairs.len());
    for &(a, b) in outlives_pairs {
        let b_s = if b == usize::MAX { "'static".to_string() } else { format!("bv{b}") };
        eprintln!("    (bv{a}, {b_s})");
    }

    eprintln!("  Augmented callee: {result_name}");
    eprintln!();
}

// ── SCC-based batch computation ───────────────────────────────────────────

/// Compute the crate-wide cast-relevant-lifetimes map using SCC-based
/// fixed-point iteration.
///
/// Runs after the collector DFS. The resulting map is stored in
/// `instance_sensitivity` for later augmentation and proxied via the
/// `cast_relevant_lifetimes` query.
///
/// # Algorithm
///
/// 1. For each collected Instance, query `items_of_instance` to obtain
///    `direct_sensitivity` and `call_sites`. Record call-graph edges.
/// 2. Compute the reverse-reachable set from directly sensitive Instances
///    to identify the sensitive subgraph.
/// 3. Compute SCCs of the sensitive subgraph (Tarjan's algorithm).
/// 4. Process SCCs in reverse topological order:
///    - Singleton SCCs (acyclic): single-pass composition.
///    - Non-trivial SCCs (cycles): iterate until fixed-point.
/// 5. Store results in `instance_sensitivity`.
pub(crate) fn compute_cast_relevant_lifetimes<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance_sensitivity: &Lock<FxIndexMap<Instance<'tcx>, InstanceSensitivity<'tcx>>>,
    visited: &UnordSet<MonoItem<'tcx>>,
) {
    // Extract Fn Instances from the visited set deterministically.
    // MonoItem implements ToStableHashKey; Instance doesn't, so we
    // sort as MonoItem first, then filter to Fn instances.
    let collected_instances: Vec<Instance<'tcx>> = tcx.with_stable_hashing_context(|mut hcx| {
        visited
            .items()
            .filter_map(|item| match item {
                MonoItem::Fn(instance) => Some(instance),
                _ => None,
            })
            .copied()
            .into_sorted(&mut hcx)
    });

    // ── Direct sensitivity + call-graph construction ──────────────────

    let mut direct_sensitivity: FxHashMap<Instance<'tcx>, CastRelevantLifetimes<'tcx>> =
        FxHashMap::default();

    let mut call_graph: FxHashMap<Instance<'tcx>, Vec<CallEdge<'tcx>>> = FxHashMap::default();

    let idx_to_inst = &collected_instances[..];
    let inst_to_idx: FxHashMap<Instance<'tcx>, u32> = collected_instances
        .iter()
        .enumerate()
        .map(|(idx, &instance)| (instance, idx as u32))
        .collect();

    // Per-instance queries in parallel: `items_of_instance` is a cacheable
    // query, and the edge/direct-sensitivity construction is read-only
    // with respect to shared state. Merge into the serial maps in input
    // order (`par_map` preserves it).
    let per_instance: Vec<(Option<CastRelevantLifetimes<'tcx>>, Vec<CallEdge<'tcx>>)> =
        par_map(collected_instances.iter().copied(), |instance| {
            let Ok(items) = tcx.items_of_instance((instance, CollectionMode::UsedItems)) else {
                return (None, Vec::new());
            };
            let direct = (!items.direct_sensitivity.is_empty())
                .then(|| CastRelevantLifetimes::from_direct_mappings(items.direct_sensitivity));
            let edges: Vec<CallEdge<'tcx>> = items
                .call_sites
                .iter()
                .filter(|(_, callee)| visited.contains(&MonoItem::Fn(*callee)))
                .map(|&(call_id, callee)| CallEdge { call_id, callee })
                .collect();
            (direct, edges)
        });

    for (&instance, (direct, edges)) in collected_instances.iter().zip(per_instance) {
        if let Some(crl) = direct {
            direct_sensitivity.insert(instance, crl);
        }
        if !edges.is_empty() {
            call_graph.insert(instance, edges);
        }
    }

    let total = idx_to_inst.len();

    // ── Sensitive subgraph (reverse reachability) ────────────────────

    let mut reverse_adj: Vec<SmallVec<[u32; 4]>> = vec![SmallVec::new(); total];
    #[allow(rustc::potential_query_instability)]
    // Iteration order doesn't matter: we're populating an adjacency list
    // indexed by pre-assigned node IDs.
    for (&caller, edges) in &call_graph {
        let caller_idx = inst_to_idx[&caller];
        for edge in edges {
            if let Some(&callee_idx) = inst_to_idx.get(&edge.callee) {
                reverse_adj[callee_idx as usize].push(caller_idx);
            }
        }
    }

    let mut sensitive_set = DenseBitSet::<u32>::new_empty(total);
    let mut queue: VecDeque<u32> = VecDeque::new();

    #[allow(rustc::potential_query_instability)]
    // Seeding a bitset by pre-assigned index; iteration order is irrelevant.
    for &inst in direct_sensitivity.keys() {
        let idx = inst_to_idx[&inst];
        if sensitive_set.insert(idx) {
            queue.push_back(idx);
        }
    }

    while let Some(idx) = queue.pop_front() {
        for &caller_idx in &reverse_adj[idx as usize] {
            if sensitive_set.insert(caller_idx) {
                queue.push_back(caller_idx);
            }
        }
    }

    if sensitive_set.is_empty() {
        return;
    }

    let n = sensitive_set.count();
    let mut full_to_sens: Vec<u32> = vec![u32::MAX; total];
    let mut sens_to_full: Vec<u32> = Vec::with_capacity(n);
    for full_idx in sensitive_set.iter() {
        let sens_idx = sens_to_full.len() as u32;
        full_to_sens[full_idx as usize] = sens_idx;
        sens_to_full.push(full_idx);
    }

    rustc_index::newtype_index! {
        #[orderable]
        struct SensIdx {}
    }
    rustc_index::newtype_index! {
        #[orderable]
        struct SccIdx {}
    }

    let mut edge_pairs: Vec<(SensIdx, SensIdx)> = Vec::new();
    for (sens_i, &full_i) in sens_to_full.iter().enumerate() {
        let full_i = full_i as usize;
        let inst = idx_to_inst[full_i];
        if let Some(edges) = call_graph.get(&inst) {
            for edge in edges {
                if let Some(&full_j) = inst_to_idx.get(&edge.callee) {
                    let sens_j = full_to_sens[full_j as usize];
                    if sens_j != u32::MAX {
                        edge_pairs.push((
                            SensIdx::from_usize(sens_i),
                            SensIdx::from_usize(sens_j as usize),
                        ));
                    }
                }
            }
        }
    }

    // ── SCC decomposition ─────────────────────────────────────────────

    let graph = VecGraph::<SensIdx, true>::new(n, edge_pairs);
    let sccs = Sccs::<SensIdx, SccIdx>::new(&graph);

    let mut scc_members: Vec<Vec<SensIdx>> = vec![Vec::new(); sccs.num_sccs()];
    for sens_i in 0..n {
        let node = SensIdx::from_usize(sens_i);
        scc_members[sccs.scc(node).index()].push(node);
    }

    // ── Pre-compute chain compositions for sensitive edges ────────────

    #[allow(rustc::potential_query_instability)]
    // Iteration order is irrelevant: populating a lookup table keyed by
    // (Instance, call_id) identity.
    let composition_cache: FxHashMap<
        (Instance<'tcx>, &'tcx List<(DefId, u32, ty::GenericArgsRef<'tcx>)>),
        Vec<Option<usize>>,
    > = {
        let mut cache = FxHashMap::default();
        for &full_i in &sens_to_full {
            let inst = idx_to_inst[full_i as usize];
            if let Some(edges) = call_graph.get(&inst) {
                for edge in edges {
                    let Some(&callee_full) = inst_to_idx.get(&edge.callee) else {
                        continue;
                    };
                    if !sensitive_set.contains(callee_full) {
                        continue;
                    }
                    // Upper bound on walk positions: total region slots
                    // in the callee's args. Positions map independently,
                    // so computing with a larger n_positions is safe.
                    let upper_bound: usize = region_slots_of_args(edge.callee.args);
                    let composed = compose_all_through_chain(tcx, inst, edge.call_id, upper_bound);
                    cache.insert((inst, edge.call_id), composed);
                }
            }
        }
        cache
    };

    // ── Process SCCs in reverse topological order ────────────────────

    let mut resolved: FxHashMap<Instance<'tcx>, CastRelevantLifetimes<'tcx>> = FxHashMap::default();

    #[allow(rustc::potential_query_instability)]
    // Seeding from pre-computed direct sensitivity; results are keyed by
    // Instance identity, not iteration order.
    for (inst, crl) in &direct_sensitivity {
        let full_idx = inst_to_idx[inst];
        if sensitive_set.contains(full_idx) {
            resolved.insert(*inst, crl.clone());
        }
    }

    for scc in sccs.all_sccs() {
        let scc: SccIdx = scc;
        let members = &scc_members[scc.index()];

        if members.len() == 1 {
            let sens_node = members[0];
            let has_self_edge = graph.successors(sens_node).contains(&sens_node);
            if !has_self_edge {
                // Singleton SCC (no self-edge): single pass.
                let instance = idx_to_inst[sens_to_full[sens_node.index()] as usize];
                let result = compute_instance_sensitivity(
                    tcx,
                    instance,
                    &call_graph,
                    &resolved,
                    &composition_cache,
                );
                if let Some(crl) = result {
                    resolved.insert(instance, crl);
                }
                continue;
            }
        }

        // Non-trivial SCC (or singleton with self-edge): iterate
        // until fixed-point.
        let member_set = {
            let mut set = DenseBitSet::<SensIdx>::new_empty(n);
            for &node in members {
                set.insert(node);
            }
            set
        };
        let mut dirty = member_set.clone();

        loop {
            let mut next_dirty = DenseBitSet::<SensIdx>::new_empty(n);
            let mut changed = false;

            for sens_node in dirty.iter() {
                let instance = idx_to_inst[sens_to_full[sens_node.index()] as usize];
                let new_crl = compute_instance_sensitivity(
                    tcx,
                    instance,
                    &call_graph,
                    &resolved,
                    &composition_cache,
                );

                if let Some(new_crl) = new_crl {
                    let (merged, changed_here) = match resolved.remove(&instance) {
                        Some(existing) => crl_join(existing, new_crl),
                        None => (new_crl, true),
                    };
                    resolved.insert(instance, merged);

                    if changed_here {
                        changed = true;
                        for &caller_sens in graph.predecessors(sens_node) {
                            if member_set.contains(caller_sens) {
                                next_dirty.insert(caller_sens);
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
            dirty = next_dirty;
        }
    }

    // ── Feed results into instance_sensitivity ───────────────────────
    //
    // Compute per-instance entries in parallel (augment_callee + arena
    // allocation are thread-safe), then insert serially into the
    // FxIndexMap. `par_map` preserves input order, so insertion order
    // remains deterministic.

    let entries: Vec<Option<(Instance<'tcx>, InstanceSensitivity<'tcx>)>> = par_map(
        collected_instances.iter().copied(),
        |instance| -> Option<(Instance<'tcx>, InstanceSensitivity<'tcx>)> {
            let sensitivity = resolved.get(&instance).cloned();
            if sensitivity.is_none() && !direct_sensitivity.contains_key(&instance) {
                return None;
            }

            let items = tcx.items_of_instance((instance, CollectionMode::UsedItems)).ok()?;

            let mut sensitive_call_sites_vec = Vec::new();
            let mut augmented_callees_vec = Vec::new();

            if sensitivity.is_some() {
                for &(call_id, callee) in items.call_sites {
                    if resolved.contains_key(&callee) {
                        sensitive_call_sites_vec.push((call_id, callee));
                    }
                }

                // For ground-level callers, pre-compute augmented callee Instances.
                if !instance.has_outlives_entries() && !sensitive_call_sites_vec.is_empty() {
                    for &(call_id, callee_instance) in &sensitive_call_sites_vec {
                        let Some(callee_sensitivity) = resolved.get(&callee_instance) else {
                            continue;
                        };

                        let composed_mapping = composition_cache
                            .get(&(instance, call_id))
                            .expect("composition cache miss during result feed-in")
                            .as_slice();
                        let caller_env = caller_env_for_call_id(tcx, instance, call_id);

                        let augmented = augment_callee(
                            tcx,
                            instance,
                            callee_instance,
                            callee_sensitivity,
                            &caller_env,
                            Some(composed_mapping),
                        );
                        augmented_callees_vec.push((call_id, augmented));
                    }
                }
            }

            Some((
                instance,
                InstanceSensitivity {
                    sensitivity,
                    sensitive_call_sites: tcx.arena.alloc_from_iter(sensitive_call_sites_vec),
                    augmented_callees: tcx.arena.alloc_from_iter(augmented_callees_vec),
                },
            ))
        },
    );

    let mut sensitivity_map = instance_sensitivity.lock();
    for (instance, sens) in entries.into_iter().flatten() {
        sensitivity_map.insert(instance, sens);
    }
    drop(sensitivity_map);

    dump_trait_cast_sensitivity(
        tcx,
        instance_sensitivity,
        &direct_sensitivity,
        &collected_instances,
    );
}

/// Dump per-instance sensitivity metadata to stderr, gated on
/// `-Z dump-trait-cast-sensitivity=<filter>`. Emits one block per matching
/// Instance in a deterministic order. Fast-paths when the flag is absent.
fn dump_trait_cast_sensitivity<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance_sensitivity: &Lock<FxIndexMap<Instance<'tcx>, InstanceSensitivity<'tcx>>>,
    direct_sensitivity: &FxHashMap<Instance<'tcx>, CastRelevantLifetimes<'tcx>>,
    _collected_instances: &[Instance<'tcx>],
) {
    let Some(ref filter) = tcx.sess.opts.unstable_opts.dump_trait_cast_sensitivity else {
        return;
    };

    let map = instance_sensitivity.lock();

    // Collect entries and sort via stable fingerprint of the Instance so the
    // output order is deterministic across runs. `Instance` is `!Ord`, so we
    // use the `with_stable_hashing_context` + `StableHasher` pattern
    // established in `cascade_canonicalize` (see partitioning.rs).
    #[allow(rustc::potential_query_instability)]
    // Collecting entries from an FxIndexMap into a Vec that we subsequently
    // sort by stable fingerprint; final iteration order is deterministic.
    let mut entries: Vec<Instance<'tcx>> = map.keys().copied().collect();
    tcx.with_stable_hashing_context(|mut hcx| {
        entries.sort_by_cached_key(|instance| {
            use rustc_data_structures::fingerprint::Fingerprint;
            use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
            let mut hasher = StableHasher::new();
            instance.hash_stable(&mut hcx, &mut hasher);
            hasher.finish::<Fingerprint>()
        });
    });

    for instance in entries {
        let entry = map.get(&instance).expect("entry present");
        let name = with_no_trimmed_paths!(instance.to_string());

        if filter != "all" && !name.contains(filter.as_str()) {
            continue;
        }

        let is_direct = direct_sensitivity.contains_key(&instance);
        let is_transitive = entry.sensitivity.is_some();
        if !is_direct && !is_transitive {
            continue;
        }

        eprintln!("=== Sensitivity: {name} ===");

        let direct_count = direct_sensitivity.get(&instance).map(|d| d.mappings.len()).unwrap_or(0);
        if is_direct {
            eprintln!("  direct:       yes  (has {direct_count} direct mapping(s))");
        } else {
            eprintln!("  direct:       no");
        }

        let transitive_count = entry.sensitivity.as_ref().map(|s| s.mappings.len()).unwrap_or(0);
        if is_transitive {
            eprintln!("  transitive:   yes  (has {transitive_count} composed mapping(s))");
        } else {
            eprintln!("  transitive:   no");
        }

        if let Some(ref sens) = entry.sensitivity {
            let sorted_mappings =
                tcx.with_stable_hashing_context(|mut hcx| sens.mappings.to_sorted(&mut hcx, true));
            eprintln!("  Mappings ({}):", sorted_mappings.len());
            for mapping in &sorted_mappings {
                let parts: Vec<String> = mapping
                    .0
                    .iter()
                    .enumerate()
                    .map(|(i, entry)| match entry {
                        Some(n) if n == usize::MAX => format!("bv{i} -> static"),
                        Some(n) => format!("bv{i} -> walk_pos={n}"),
                        None => format!("bv{i} -> (none)"),
                    })
                    .collect();
                eprintln!("    [{}]", parts.join(", "));
            }
        }

        eprintln!("  Sensitive call sites ({}):", entry.sensitive_call_sites.len());
        for &(call_id, callee) in entry.sensitive_call_sites {
            let summary = format_call_id_summary(tcx, call_id);
            let callee_str = with_no_trimmed_paths!(callee.to_string());
            eprintln!("    {summary} -> {callee_str}");
        }

        if !entry.augmented_callees.is_empty() {
            eprintln!("  Augmented callees ({}):", entry.augmented_callees.len());
            for &(call_id, callee) in entry.augmented_callees {
                let summary = format_call_id_summary(tcx, call_id);
                let callee_str = with_no_trimmed_paths!(callee.to_string());
                eprintln!("    {summary} -> {callee_str}");
            }
        }

        eprintln!();
    }
}

/// Render a `call_id` chain as `call#<local_id> in <def_path>` using the
/// first element (origin call site). If the chain is inlined (length > 1),
/// annotate with `(+N more)`.
pub(crate) fn format_call_id_summary<'tcx>(
    tcx: TyCtxt<'tcx>,
    call_id: &'tcx List<(DefId, u32, ty::GenericArgsRef<'tcx>)>,
) -> String {
    let Some((origin_def_id, local_id, _)) = call_id.iter().next() else {
        return "call#(empty-chain)".to_string();
    };
    let def_path = tcx.def_path_str(origin_def_id);
    let extra = call_id.len().saturating_sub(1);
    if extra == 0 {
        format!("call#{local_id} in {def_path}")
    } else {
        format!("call#{local_id} in {def_path} (+{extra} more)")
    }
}

/// Compute transitive sensitivity for a single Instance by examining
/// its callees' current sensitivity state.
fn compute_instance_sensitivity<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    call_graph: &FxHashMap<Instance<'tcx>, Vec<CallEdge<'tcx>>>,
    resolved: &FxHashMap<Instance<'tcx>, CastRelevantLifetimes<'tcx>>,
    composition_cache: &FxHashMap<
        (Instance<'tcx>, &'tcx List<(DefId, u32, ty::GenericArgsRef<'tcx>)>),
        Vec<Option<usize>>,
    >,
) -> Option<CastRelevantLifetimes<'tcx>> {
    let edges = match call_graph.get(&instance) {
        Some(e) => e,
        None => return resolved.get(&instance).cloned(),
    };

    let mut composed_mappings: UnordSet<LifetimeBVToParamMapping<'tcx>> = UnordSet::new();

    for edge in edges {
        let Some(callee_sensitivity) = resolved.get(&edge.callee) else {
            continue;
        };

        let composed_positions = composition_cache
            .get(&(instance, edge.call_id))
            .expect("composition cache miss in compute_instance_sensitivity")
            .as_slice();
        composed_mappings.extend_unord(callee_sensitivity.mappings.items().filter_map(|mapping| {
            let composed_bv = mapping.0.iter().map(|entry| {
                entry.and_then(|walk_pos| composed_positions.get(walk_pos).copied().flatten())
            });
            if composed_bv.clone().any(|b| b.is_some()) {
                let list = tcx.mk_lifetime_bv_to_param_mapping_from_iter(composed_bv);
                Some(LifetimeBVToParamMapping(list))
            } else {
                None
            }
        }));
    }

    // Merge with direct sensitivity if present.
    if let Some(direct) = resolved.get(&instance) {
        if composed_mappings.is_empty() {
            return Some(direct.clone());
        }
        composed_mappings.extend_unord(direct.mappings.items().copied());
        return Some(CastRelevantLifetimes { mappings: composed_mappings });
    }

    if composed_mappings.is_empty() {
        return None;
    }
    Some(CastRelevantLifetimes { mappings: composed_mappings })
}

/// Monotone join: merges mappings from `b` into `a`. Returns
/// `(merged, changed)` where `changed` is true iff at least one new
/// mapping was inserted. O(1) per-element duplicate detection via
/// `UnordSet::insert`.
fn crl_join<'tcx>(
    mut a: CastRelevantLifetimes<'tcx>,
    b: CastRelevantLifetimes<'tcx>,
) -> (CastRelevantLifetimes<'tcx>, bool) {
    let before = a.mappings.len();
    a.mappings.extend_unord(b.mappings.into_items());
    let changed = a.mappings.len() > before;
    (a, changed)
}

// ── augmented_outlives_for_call query provider ────────────────────────────

/// Query provider for `augmented_outlives_for_call`.
///
/// For MIR-backed callees, computes per-call-site outlives entries in the
/// callee's binder-variable index space by composing the callee's
/// `CastRelevantLifetimes` through the `call_id` chain and consulting the
/// caller's outlives environment.
///
/// For MIR-less intrinsic callees, the fallback path returns entries in the
/// origin/transport walk-position space induced by the caller's own CRL;
/// intrinsic-specific resolvers remap those entries into their native
/// consumer space before interpreting them.
pub(crate) fn augmented_outlives_for_call<'tcx>(
    tcx: TyCtxt<'tcx>,
    (caller, call_id, callee): (
        Instance<'tcx>,
        &'tcx List<(DefId, u32, ty::GenericArgsRef<'tcx>)>,
        Instance<'tcx>,
    ),
) -> &'tcx [ty::GenericArg<'tcx>] {
    // 1. Get callee's cast-relevant lifetimes.
    let Some(callee_sensitivity) = tcx.cast_relevant_lifetimes(callee) else {
        // Intrinsic callees (trait_metadata_index,
        // trait_cast_is_lifetime_erasure_safe, etc.) are never in the
        // sensitivity map — they have no MIR body. For these, use the
        // caller's direct-sensitivity mapping if the caller is already
        // augmented, or fall back to the intrinsic call site's identity
        // walk-position mapping for true ground-level callers.
        let callee_def_id = callee.def_id();
        if tcx.is_intrinsic(callee_def_id, sym::trait_metadata_index)
            || tcx.is_intrinsic(callee_def_id, sym::trait_cast_is_lifetime_erasure_safe)
            || tcx.is_intrinsic(callee_def_id, sym::trait_metadata_table)
            || tcx.is_intrinsic(callee_def_id, sym::trait_metadata_table_len)
        {
            // Sentinel-only augmented instances (only the sentinel,
            // no real outlives pairs) have no outlives environment —
            // fall through to the ground-level path that uses the
            // borrowck region summary.
            let has_real_outlives = caller.outlives_entries().len() > 1;
            let augmented = if has_real_outlives {
                let caller_base = caller.strip_outlives(tcx);
                let Ok(items) = tcx.items_of_instance((caller_base, CollectionMode::UsedItems))
                else {
                    return &[];
                };
                if items.direct_sensitivity.is_empty() {
                    return &[];
                }
                let caller_direct =
                    CastRelevantLifetimes::from_direct_mappings(items.direct_sensitivity);
                let caller_env = caller_env_for_call_id(tcx, caller, call_id);
                augment_callee(tcx, caller, callee, &caller_direct, &caller_env, None)
            } else {
                let origin_def_id = call_id[0].0;
                let origin_local_id = call_id[0].1;
                let summary = tcx.borrowck_region_summary(origin_def_id);
                let Some(mapping) = summary.call_site_mappings.get(&origin_local_id) else {
                    return &[];
                };
                let Some(identity_sensitivity) =
                    input_identity_sensitivity_for_call_site(tcx, summary, mapping)
                else {
                    return &[];
                };
                let caller_env =
                    CallerOutlivesEnv::from_region_summary_walk_pos(tcx, summary, mapping);
                augment_callee(tcx, caller, callee, &identity_sensitivity, &caller_env, None)
            };
            let all = augmented.outlives_entries();
            if all.len() > 1 {
                return tcx.arena.alloc_slice(&all[1..]);
            }
        }
        return &[];
    };

    // 2. Build composed_mapping: maps callee walk-order positions to
    //    the origin caller's walk-position space by composing through
    //    the full call_id chain in one pass.
    let max_pos = callee_sensitivity.max_walk_order_position();
    let composed_mapping = compose_all_through_chain(tcx, caller, call_id, max_pos);

    // 3. Build CallerOutlivesEnv from the caller Instance.
    let caller_env = caller_env_for_call_id(tcx, caller, call_id);
    // 4. Execute augment_callee.
    let augmented = augment_callee(
        tcx,
        caller,
        callee,
        callee_sensitivity,
        &caller_env,
        Some(&composed_mapping),
    );

    // 5. Return the outlives entries, stripping the sentinel (always at
    //    position 0 of outlives_entries).
    let all = augmented.outlives_entries();
    if all.len() <= 1 {
        // Empty or only the sentinel — no meaningful outlives entries.
        return &[];
    }
    // Skip sentinel at index 0.
    tcx.arena.alloc_slice(&all[1..])
}

// ── Build sensitivity_map for MonoItemPartitions ──────────────────────────

/// Convert the internal `instance_sensitivity` map into the `UnordMap`
/// used by `MonoItemPartitions::sensitivity_map` and the
/// `crate_cast_relevant_lifetimes` query.
pub(crate) fn build_sensitivity_map<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance_sensitivity: FxIndexMap<Instance<'tcx>, InstanceSensitivity<'tcx>>,
) -> &'tcx UnordMap<Instance<'tcx>, CastRelevantLifetimes<'tcx>> {
    let map: UnordMap<_, _> = instance_sensitivity
        .into_iter()
        .filter_map(|(inst, entry)| entry.sensitivity.map(|crl| (inst, crl)))
        .collect();
    tcx.arena.alloc(map)
}
