use rustc_hir::attrs::{AttributeKind, CoverageAttrKind};
use rustc_hir::find_attr;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::coverage::{BasicCoverageBlock, CoverageIdsInfo, CoverageKind, MappingKind};
use rustc_middle::mir::{Body, Statement, StatementKind};
use rustc_middle::ty::{self, TyCtxt};
use rustc_middle::util::Providers;
use rustc_span::def_id::LocalDefId;
use tracing::trace;

use crate::coverage::counters::node_flow::make_node_counters;
use crate::coverage::counters::{CoverageCounters, transcribe_counters};

/// Registers query/hook implementations related to coverage.
pub(crate) fn provide(providers: &mut Providers) {
    providers.hooks.is_eligible_for_coverage = is_eligible_for_coverage;
    providers.queries.coverage_attr_on = coverage_attr_on;
    providers.queries.coverage_ids_info = coverage_ids_info;
}

/// Hook implementation for [`TyCtxt::is_eligible_for_coverage`].
fn is_eligible_for_coverage(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    // Only instrument functions, methods, and closures (not constants since they are evaluated
    // at compile time by Miri).
    // FIXME(#73156): Handle source code coverage in const eval, but note, if and when const
    // expressions get coverage spans, we will probably have to "carve out" space for const
    // expressions from coverage spans in enclosing MIR's, like we do for closures. (That might
    // be tricky if const expressions have no corresponding statements in the enclosing MIR.
    // Closures are carved out by their initial `Assign` statement.)
    if !tcx.def_kind(def_id).is_fn_like() {
        trace!("InstrumentCoverage skipped for {def_id:?} (not an fn-like)");
        return false;
    }

    if tcx.codegen_fn_attrs(def_id).flags.contains(CodegenFnAttrFlags::NAKED) {
        trace!("InstrumentCoverage skipped for {def_id:?} (`#[naked]`)");
        return false;
    }

    if !tcx.coverage_attr_on(def_id) {
        trace!("InstrumentCoverage skipped for {def_id:?} (`#[coverage(off)]`)");
        return false;
    }

    true
}

/// Query implementation for `coverage_attr_on`.
fn coverage_attr_on(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    // Check for a `#[coverage(..)]` attribute on this def.
    if let Some(kind) =
        find_attr!(tcx.get_all_attrs(def_id), AttributeKind::Coverage(_sp, kind) => kind)
    {
        match kind {
            CoverageAttrKind::On => return true,
            CoverageAttrKind::Off => return false,
        }
    };

    // Treat `#[automatically_derived]` as an implied `#[coverage(off)]`, on
    // the assumption that most users won't want coverage for derived impls.
    //
    // This affects not just the associated items of an impl block, but also
    // any closures and other nested functions within those associated items.
    if tcx.is_automatically_derived(def_id.to_def_id()) {
        return false;
    }

    // Check the parent def (and so on recursively) until we find an
    // enclosing attribute or reach the crate root.
    match tcx.opt_local_parent(def_id) {
        Some(parent) => tcx.coverage_attr_on(parent),
        // We reached the crate root without seeing a coverage attribute, so
        // allow coverage instrumentation by default.
        None => true,
    }
}

/// Query implementation for `coverage_ids_info`.
fn coverage_ids_info<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance_def: ty::InstanceKind<'tcx>,
) -> Option<CoverageIdsInfo> {
    let mir_body = tcx.instance_mir(instance_def);
    let fn_cov_info = mir_body.function_coverage_info.as_deref()?;

    // Scan through the final MIR to see which BCBs survived MIR opts.
    // Any BCB not in this set was optimized away.
    let mut bcbs_seen = DenseBitSet::new_empty(fn_cov_info.priority_list.len());
    for kind in all_coverage_in_mir_body(mir_body) {
        match *kind {
            CoverageKind::VirtualCounter { bcb } => {
                bcbs_seen.insert(bcb);
            }
            _ => {}
        }
    }

    // Determine the set of BCBs that are referred to by mappings, and therefore
    // need a counter. Any node not in this set will only get a counter if it
    // is part of the counter expression for a node that is in the set.
    let mut bcb_needs_counter =
        DenseBitSet::<BasicCoverageBlock>::new_empty(fn_cov_info.priority_list.len());
    for mapping in &fn_cov_info.mappings {
        match mapping.kind {
            MappingKind::Code { bcb } => {
                bcb_needs_counter.insert(bcb);
            }
            MappingKind::Branch { true_bcb, false_bcb } => {
                bcb_needs_counter.insert(true_bcb);
                bcb_needs_counter.insert(false_bcb);
            }
        }
    }

    // Clone the priority list so that we can re-sort it.
    let mut priority_list = fn_cov_info.priority_list.clone();
    // The first ID in the priority list represents the synthetic "sink" node,
    // and must remain first so that it _never_ gets a physical counter.
    debug_assert_eq!(priority_list[0], priority_list.iter().copied().max().unwrap());
    assert!(!bcbs_seen.contains(priority_list[0]));
    // Partition the priority list, so that unreachable nodes (removed by MIR opts)
    // are sorted later and therefore are _more_ likely to get a physical counter.
    // This is counter-intuitive, but it means that `transcribe_counters` can
    // easily skip those unused physical counters and replace them with zero.
    // (The original ordering remains in effect within both partitions.)
    priority_list[1..].sort_by_key(|&bcb| !bcbs_seen.contains(bcb));

    let node_counters = make_node_counters(&fn_cov_info.node_flow_data, &priority_list);
    let coverage_counters = transcribe_counters(&node_counters, &bcb_needs_counter, &bcbs_seen);

    let CoverageCounters {
        phys_counter_for_node, next_counter_id, node_counters, expressions, ..
    } = coverage_counters;

    Some(CoverageIdsInfo {
        num_counters: next_counter_id.as_u32(),
        phys_counter_for_node,
        term_for_bcb: node_counters,
        expressions,
    })
}

fn all_coverage_in_mir_body<'a, 'tcx>(
    body: &'a Body<'tcx>,
) -> impl Iterator<Item = &'a CoverageKind> {
    body.basic_blocks.iter().flat_map(|bb_data| &bb_data.statements).filter_map(|statement| {
        match statement.kind {
            StatementKind::Coverage(ref kind) if !is_inlined(body, statement) => Some(kind),
            _ => None,
        }
    })
}

fn is_inlined(body: &Body<'_>, statement: &Statement<'_>) -> bool {
    let scope_data = &body.source_scopes[statement.source_info.scope];
    scope_data.inlined.is_some() || scope_data.inlined_parent_scope.is_some()
}
