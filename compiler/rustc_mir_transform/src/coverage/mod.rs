pub(super) mod query;

mod counters;
mod graph;
mod mappings;
mod spans;
#[cfg(test)]
mod tests;
mod unexpand;

use rustc_hir as hir;
use rustc_hir::intravisit::{Visitor, walk_expr};
use rustc_middle::hir::map::Map;
use rustc_middle::hir::nested_filter;
use rustc_middle::mir::coverage::{
    CoverageKind, DecisionInfo, FunctionCoverageInfo, Mapping, MappingKind, SourceRegion,
};
use rustc_middle::mir::{
    self, BasicBlock, BasicBlockData, SourceInfo, Statement, StatementKind, Terminator,
    TerminatorKind,
};
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::LocalDefId;
use rustc_span::source_map::SourceMap;
use rustc_span::{BytePos, Pos, SourceFile, Span};
use tracing::{debug, debug_span, trace};

use crate::coverage::counters::{CoverageCounters, Site};
use crate::coverage::graph::CoverageGraph;
use crate::coverage::mappings::ExtractedMappings;

/// Inserts `StatementKind::Coverage` statements that either instrument the binary with injected
/// counters, via intrinsic `llvm.instrprof.increment`, and/or inject metadata used during codegen
/// to construct the coverage map.
pub(super) struct InstrumentCoverage;

impl<'tcx> crate::MirPass<'tcx> for InstrumentCoverage {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.instrument_coverage()
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, mir_body: &mut mir::Body<'tcx>) {
        let mir_source = mir_body.source;

        // This pass runs after MIR promotion, but before promoted MIR starts to
        // be transformed, so it should never see promoted MIR.
        assert!(mir_source.promoted.is_none());

        let def_id = mir_source.def_id().expect_local();

        if !tcx.is_eligible_for_coverage(def_id) {
            trace!("InstrumentCoverage skipped for {def_id:?} (not eligible)");
            return;
        }

        // An otherwise-eligible function is still skipped if its start block
        // is known to be unreachable.
        match mir_body.basic_blocks[mir::START_BLOCK].terminator().kind {
            TerminatorKind::Unreachable => {
                trace!("InstrumentCoverage skipped for unreachable `START_BLOCK`");
                return;
            }
            _ => {}
        }

        instrument_function_for_coverage(tcx, mir_body);
    }
}

fn instrument_function_for_coverage<'tcx>(tcx: TyCtxt<'tcx>, mir_body: &mut mir::Body<'tcx>) {
    let def_id = mir_body.source.def_id();
    let _span = debug_span!("instrument_function_for_coverage", ?def_id).entered();

    let hir_info = extract_hir_info(tcx, def_id.expect_local());
    let basic_coverage_blocks = CoverageGraph::from_mir(mir_body);

    ////////////////////////////////////////////////////
    // Extract coverage spans and other mapping info from MIR.
    let extracted_mappings = mappings::extract_all_mapping_info_from_mir(
        tcx,
        mir_body,
        &hir_info,
        &basic_coverage_blocks,
    );

    ////////////////////////////////////////////////////
    // Create an optimized mix of `Counter`s and `Expression`s for the `CoverageGraph`. Ensure
    // every coverage span has a `Counter` or `Expression` assigned to its `BasicCoverageBlock`
    // and all `Expression` dependencies (operands) are also generated, for any other
    // `BasicCoverageBlock`s not already associated with a coverage span.
    let bcbs_with_counter_mappings = extracted_mappings.all_bcbs_with_counter_mappings();
    if bcbs_with_counter_mappings.is_empty() {
        // No relevant spans were found in MIR, so skip instrumenting this function.
        return;
    }

    let coverage_counters =
        CoverageCounters::make_bcb_counters(&basic_coverage_blocks, &bcbs_with_counter_mappings);

    let mappings = create_mappings(tcx, &hir_info, &extracted_mappings, &coverage_counters);
    if mappings.is_empty() {
        // No spans could be converted into valid mappings, so skip this function.
        debug!("no spans could be converted into valid mappings; skipping");
        return;
    }

    inject_coverage_statements(
        mir_body,
        &basic_coverage_blocks,
        &extracted_mappings,
        &coverage_counters,
    );

    inject_mcdc_statements(mir_body, &basic_coverage_blocks, &extracted_mappings);

    let mcdc_num_condition_bitmaps = extracted_mappings
        .mcdc_mappings
        .iter()
        .map(|&(mappings::MCDCDecision { decision_depth, .. }, _)| decision_depth)
        .max()
        .map_or(0, |max| usize::from(max) + 1);

    mir_body.function_coverage_info = Some(Box::new(FunctionCoverageInfo {
        function_source_hash: hir_info.function_source_hash,
        body_span: hir_info.body_span,
        num_counters: coverage_counters.num_counters(),
        mcdc_bitmap_bits: extracted_mappings.mcdc_bitmap_bits,
        expressions: coverage_counters.into_expressions(),
        mappings,
        mcdc_num_condition_bitmaps,
    }));
}

/// For each coverage span extracted from MIR, create a corresponding
/// mapping.
///
/// Precondition: All BCBs corresponding to those spans have been given
/// coverage counters.
fn create_mappings<'tcx>(
    tcx: TyCtxt<'tcx>,
    hir_info: &ExtractedHirInfo,
    extracted_mappings: &ExtractedMappings,
    coverage_counters: &CoverageCounters,
) -> Vec<Mapping> {
    let source_map = tcx.sess.source_map();
    let file = source_map.lookup_source_file(hir_info.body_span.lo());

    let term_for_bcb =
        |bcb| coverage_counters.term_for_bcb(bcb).expect("all BCBs with spans were given counters");
    let region_for_span = |span: Span| make_source_region(source_map, hir_info, &file, span);

    // Fully destructure the mappings struct to make sure we don't miss any kinds.
    let ExtractedMappings {
        num_bcbs: _,
        code_mappings,
        branch_pairs,
        mcdc_bitmap_bits: _,
        mcdc_degraded_branches,
        mcdc_mappings,
    } = extracted_mappings;
    let mut mappings = Vec::new();

    mappings.extend(code_mappings.iter().filter_map(
        // Ordinary code mappings are the simplest kind.
        |&mappings::CodeMapping { span, bcb }| {
            let source_region = region_for_span(span)?;
            let kind = MappingKind::Code(term_for_bcb(bcb));
            Some(Mapping { kind, source_region })
        },
    ));

    mappings.extend(branch_pairs.iter().filter_map(
        |&mappings::BranchPair { span, true_bcb, false_bcb }| {
            let true_term = term_for_bcb(true_bcb);
            let false_term = term_for_bcb(false_bcb);
            let kind = MappingKind::Branch { true_term, false_term };
            let source_region = region_for_span(span)?;
            Some(Mapping { kind, source_region })
        },
    ));

    let term_for_bcb =
        |bcb| coverage_counters.term_for_bcb(bcb).expect("all BCBs with spans were given counters");

    // MCDC branch mappings are appended with their decisions in case decisions were ignored.
    mappings.extend(mcdc_degraded_branches.iter().filter_map(
        |&mappings::MCDCBranch {
             span,
             true_bcb,
             false_bcb,
             condition_info: _,
             true_index: _,
             false_index: _,
         }| {
            let source_region = region_for_span(span)?;
            let true_term = term_for_bcb(true_bcb);
            let false_term = term_for_bcb(false_bcb);
            Some(Mapping { kind: MappingKind::Branch { true_term, false_term }, source_region })
        },
    ));

    for (decision, branches) in mcdc_mappings {
        let num_conditions = branches.len() as u16;
        let conditions = branches
            .into_iter()
            .filter_map(
                |&mappings::MCDCBranch {
                     span,
                     true_bcb,
                     false_bcb,
                     condition_info,
                     true_index: _,
                     false_index: _,
                 }| {
                    let source_region = region_for_span(span)?;
                    let true_term = term_for_bcb(true_bcb);
                    let false_term = term_for_bcb(false_bcb);
                    Some(Mapping {
                        kind: MappingKind::MCDCBranch {
                            true_term,
                            false_term,
                            mcdc_params: condition_info,
                        },
                        source_region,
                    })
                },
            )
            .collect::<Vec<_>>();

        if conditions.len() == num_conditions as usize
            && let Some(source_region) = region_for_span(decision.span)
        {
            // LLVM requires end index for counter mapping regions.
            let kind = MappingKind::MCDCDecision(DecisionInfo {
                bitmap_idx: (decision.bitmap_idx + decision.num_test_vectors) as u32,
                num_conditions,
            });
            mappings.extend(
                std::iter::once(Mapping { kind, source_region }).chain(conditions.into_iter()),
            );
        } else {
            mappings.extend(conditions.into_iter().map(|mapping| {
                let MappingKind::MCDCBranch { true_term, false_term, mcdc_params: _ } =
                    mapping.kind
                else {
                    unreachable!("all mappings here are MCDCBranch as shown above");
                };
                Mapping {
                    kind: MappingKind::Branch { true_term, false_term },
                    source_region: mapping.source_region,
                }
            }))
        }
    }

    mappings
}

/// For each BCB node or BCB edge that has an associated coverage counter,
/// inject any necessary coverage statements into MIR.
fn inject_coverage_statements<'tcx>(
    mir_body: &mut mir::Body<'tcx>,
    basic_coverage_blocks: &CoverageGraph,
    extracted_mappings: &ExtractedMappings,
    coverage_counters: &CoverageCounters,
) {
    // Inject counter-increment statements into MIR.
    for (id, site) in coverage_counters.counter_increment_sites() {
        // Determine the block to inject a counter-increment statement into.
        // For BCB nodes this is just their first block, but for edges we need
        // to create a new block between the two BCBs, and inject into that.
        let target_bb = match site {
            Site::Node { bcb } => basic_coverage_blocks[bcb].leader_bb(),
            Site::Edge { from_bcb, to_bcb } => {
                // Create a new block between the last block of `from_bcb` and
                // the first block of `to_bcb`.
                let from_bb = basic_coverage_blocks[from_bcb].last_bb();
                let to_bb = basic_coverage_blocks[to_bcb].leader_bb();

                let new_bb = inject_edge_counter_basic_block(mir_body, from_bb, to_bb);
                debug!(
                    "Edge {from_bcb:?} (last {from_bb:?}) -> {to_bcb:?} (leader {to_bb:?}) \
                    requires a new MIR BasicBlock {new_bb:?} for counter increment {id:?}",
                );
                new_bb
            }
        };

        inject_statement(mir_body, CoverageKind::CounterIncrement { id }, target_bb);
    }

    // For each counter expression that is directly associated with at least one
    // span, we inject an "expression-used" statement, so that coverage codegen
    // can check whether the injected statement survived MIR optimization.
    // (BCB edges can't have spans, so we only need to process BCB nodes here.)
    //
    // We only do this for ordinary `Code` mappings, because branch and MC/DC
    // mappings might have expressions that don't correspond to any single
    // point in the control-flow graph.
    //
    // See the code in `rustc_codegen_llvm::coverageinfo::map_data` that deals
    // with "expressions seen" and "zero terms".
    let eligible_bcbs = extracted_mappings.bcbs_with_ordinary_code_mappings();
    for (bcb, expression_id) in coverage_counters
        .bcb_nodes_with_coverage_expressions()
        .filter(|&(bcb, _)| eligible_bcbs.contains(bcb))
    {
        inject_statement(
            mir_body,
            CoverageKind::ExpressionUsed { id: expression_id },
            basic_coverage_blocks[bcb].leader_bb(),
        );
    }
}

/// For each conditions inject statements to update condition bitmap after it has been evaluated.
/// For each decision inject statements to update test vector bitmap after it has been evaluated.
fn inject_mcdc_statements<'tcx>(
    mir_body: &mut mir::Body<'tcx>,
    basic_coverage_blocks: &CoverageGraph,
    extracted_mappings: &ExtractedMappings,
) {
    for (decision, conditions) in &extracted_mappings.mcdc_mappings {
        // Inject test vector update first because `inject_statement` always insert new statement at head.
        for &end in &decision.end_bcbs {
            let end_bb = basic_coverage_blocks[end].leader_bb();
            inject_statement(
                mir_body,
                CoverageKind::TestVectorBitmapUpdate {
                    bitmap_idx: decision.bitmap_idx as u32,
                    decision_depth: decision.decision_depth,
                },
                end_bb,
            );
        }

        for &mappings::MCDCBranch {
            span: _,
            true_bcb,
            false_bcb,
            condition_info: _,
            true_index,
            false_index,
        } in conditions
        {
            for (index, bcb) in [(false_index, false_bcb), (true_index, true_bcb)] {
                let bb = basic_coverage_blocks[bcb].leader_bb();
                inject_statement(
                    mir_body,
                    CoverageKind::CondBitmapUpdate {
                        index: index as u32,
                        decision_depth: decision.decision_depth,
                    },
                    bb,
                );
            }
        }
    }
}

/// Given two basic blocks that have a control-flow edge between them, creates
/// and returns a new block that sits between those blocks.
fn inject_edge_counter_basic_block(
    mir_body: &mut mir::Body<'_>,
    from_bb: BasicBlock,
    to_bb: BasicBlock,
) -> BasicBlock {
    let span = mir_body[from_bb].terminator().source_info.span.shrink_to_hi();
    let new_bb = mir_body.basic_blocks_mut().push(BasicBlockData {
        statements: vec![], // counter will be injected here
        terminator: Some(Terminator {
            source_info: SourceInfo::outermost(span),
            kind: TerminatorKind::Goto { target: to_bb },
        }),
        is_cleanup: false,
    });
    let edge_ref = mir_body[from_bb]
        .terminator_mut()
        .successors_mut()
        .find(|successor| **successor == to_bb)
        .expect("from_bb should have a successor for to_bb");
    *edge_ref = new_bb;
    new_bb
}

fn inject_statement(mir_body: &mut mir::Body<'_>, counter_kind: CoverageKind, bb: BasicBlock) {
    debug!("  injecting statement {counter_kind:?} for {bb:?}");
    let data = &mut mir_body[bb];
    let source_info = data.terminator().source_info;
    let statement = Statement { source_info, kind: StatementKind::Coverage(counter_kind) };
    data.statements.insert(0, statement);
}

fn ensure_non_empty_span(
    source_map: &SourceMap,
    hir_info: &ExtractedHirInfo,
    span: Span,
) -> Option<Span> {
    if !span.is_empty() {
        return Some(span);
    }

    let lo = span.lo();
    let hi = span.hi();

    // The span is empty, so try to expand it to cover an adjacent '{' or '}',
    // but only within the bounds of the body span.
    let try_next = hi < hir_info.body_span.hi();
    let try_prev = hir_info.body_span.lo() < lo;
    if !(try_next || try_prev) {
        return None;
    }

    source_map
        .span_to_source(span, |src, start, end| try {
            // We're only checking for specific ASCII characters, so we don't
            // have to worry about multi-byte code points.
            if try_next && src.as_bytes()[end] == b'{' {
                Some(span.with_hi(hi + BytePos(1)))
            } else if try_prev && src.as_bytes()[start - 1] == b'}' {
                Some(span.with_lo(lo - BytePos(1)))
            } else {
                None
            }
        })
        .ok()?
}

/// Converts the span into its start line and column, and end line and column.
///
/// Line numbers and column numbers are 1-based. Unlike most column numbers emitted by
/// the compiler, these column numbers are denoted in **bytes**, because that's what
/// LLVM's `llvm-cov` tool expects to see in coverage maps.
///
/// Returns `None` if the conversion failed for some reason. This shouldn't happen,
/// but it's hard to rule out entirely (especially in the presence of complex macros
/// or other expansions), and if it does happen then skipping a span or function is
/// better than an ICE or `llvm-cov` failure that the user might have no way to avoid.
fn make_source_region(
    source_map: &SourceMap,
    hir_info: &ExtractedHirInfo,
    file: &SourceFile,
    span: Span,
) -> Option<SourceRegion> {
    let span = ensure_non_empty_span(source_map, hir_info, span)?;

    let lo = span.lo();
    let hi = span.hi();

    // Column numbers need to be in bytes, so we can't use the more convenient
    // `SourceMap` methods for looking up file coordinates.
    let line_and_byte_column = |pos: BytePos| -> Option<(usize, usize)> {
        let rpos = file.relative_position(pos);
        let line_index = file.lookup_line(rpos)?;
        let line_start = file.lines()[line_index];
        // Line numbers and column numbers are 1-based, so add 1 to each.
        Some((line_index + 1, (rpos - line_start).to_usize() + 1))
    };

    let (mut start_line, start_col) = line_and_byte_column(lo)?;
    let (mut end_line, end_col) = line_and_byte_column(hi)?;

    // Apply an offset so that code in doctests has correct line numbers.
    // FIXME(#79417): Currently we have no way to offset doctest _columns_.
    start_line = source_map.doctest_offset_line(&file.name, start_line);
    end_line = source_map.doctest_offset_line(&file.name, end_line);

    check_source_region(SourceRegion {
        start_line: start_line as u32,
        start_col: start_col as u32,
        end_line: end_line as u32,
        end_col: end_col as u32,
    })
}

/// If `llvm-cov` sees a source region that is improperly ordered (end < start),
/// it will immediately exit with a fatal error. To prevent that from happening,
/// discard regions that are improperly ordered, or might be interpreted in a
/// way that makes them improperly ordered.
fn check_source_region(source_region: SourceRegion) -> Option<SourceRegion> {
    let SourceRegion { start_line, start_col, end_line, end_col } = source_region;

    // Line/column coordinates are supposed to be 1-based. If we ever emit
    // coordinates of 0, `llvm-cov` might misinterpret them.
    let all_nonzero = [start_line, start_col, end_line, end_col].into_iter().all(|x| x != 0);
    // Coverage mappings use the high bit of `end_col` to indicate that a
    // region is actually a "gap" region, so make sure it's unset.
    let end_col_has_high_bit_unset = (end_col & (1 << 31)) == 0;
    // If a region is improperly ordered (end < start), `llvm-cov` will exit
    // with a fatal error, which is inconvenient for users and hard to debug.
    let is_ordered = (start_line, start_col) <= (end_line, end_col);

    if all_nonzero && end_col_has_high_bit_unset && is_ordered {
        Some(source_region)
    } else {
        debug!(
            ?source_region,
            ?all_nonzero,
            ?end_col_has_high_bit_unset,
            ?is_ordered,
            "Skipping source region that would be misinterpreted or rejected by LLVM"
        );
        // If this happens in a debug build, ICE to make it easier to notice.
        debug_assert!(false, "Improper source region: {source_region:?}");
        None
    }
}

/// Function information extracted from HIR by the coverage instrumentor.
#[derive(Debug)]
struct ExtractedHirInfo {
    function_source_hash: u64,
    is_async_fn: bool,
    /// The span of the function's signature, extended to the start of `body_span`.
    /// Must have the same context and filename as the body span.
    fn_sig_span_extended: Option<Span>,
    body_span: Span,
    /// "Holes" are regions within the body span that should not be included in
    /// coverage spans for this function (e.g. closures and nested items).
    hole_spans: Vec<Span>,
}

fn extract_hir_info<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> ExtractedHirInfo {
    // FIXME(#79625): Consider improving MIR to provide the information needed, to avoid going back
    // to HIR for it.

    // HACK: For synthetic MIR bodies (async closures), use the def id of the HIR body.
    if tcx.is_synthetic_mir(def_id) {
        return extract_hir_info(tcx, tcx.local_parent(def_id));
    }

    let hir_node = tcx.hir_node_by_def_id(def_id);
    let fn_body_id = hir_node.body_id().expect("HIR node is a function with body");
    let hir_body = tcx.hir().body(fn_body_id);

    let maybe_fn_sig = hir_node.fn_sig();
    let is_async_fn = maybe_fn_sig.is_some_and(|fn_sig| fn_sig.header.is_async());

    let mut body_span = hir_body.value.span;

    use hir::{Closure, Expr, ExprKind, Node};
    // Unexpand a closure's body span back to the context of its declaration.
    // This helps with closure bodies that consist of just a single bang-macro,
    // and also with closure bodies produced by async desugaring.
    if let Node::Expr(&Expr { kind: ExprKind::Closure(&Closure { fn_decl_span, .. }), .. }) =
        hir_node
    {
        body_span = body_span.find_ancestor_in_same_ctxt(fn_decl_span).unwrap_or(body_span);
    }

    // The actual signature span is only used if it has the same context and
    // filename as the body, and precedes the body.
    let fn_sig_span_extended = maybe_fn_sig
        .map(|fn_sig| fn_sig.span)
        .filter(|&fn_sig_span| {
            let source_map = tcx.sess.source_map();
            let file_idx = |span: Span| source_map.lookup_source_file_idx(span.lo());

            fn_sig_span.eq_ctxt(body_span)
                && fn_sig_span.hi() <= body_span.lo()
                && file_idx(fn_sig_span) == file_idx(body_span)
        })
        // If so, extend it to the start of the body span.
        .map(|fn_sig_span| fn_sig_span.with_hi(body_span.lo()));

    let function_source_hash = hash_mir_source(tcx, hir_body);

    let hole_spans = extract_hole_spans_from_hir(tcx, body_span, hir_body);

    ExtractedHirInfo {
        function_source_hash,
        is_async_fn,
        fn_sig_span_extended,
        body_span,
        hole_spans,
    }
}

fn hash_mir_source<'tcx>(tcx: TyCtxt<'tcx>, hir_body: &'tcx hir::Body<'tcx>) -> u64 {
    // FIXME(cjgillot) Stop hashing HIR manually here.
    let owner = hir_body.id().hir_id.owner;
    tcx.hir_owner_nodes(owner).opt_hash_including_bodies.unwrap().to_smaller_hash().as_u64()
}

fn extract_hole_spans_from_hir<'tcx>(
    tcx: TyCtxt<'tcx>,
    body_span: Span, // Usually `hir_body.value.span`, but not always
    hir_body: &hir::Body<'tcx>,
) -> Vec<Span> {
    struct HolesVisitor<'hir, F> {
        hir: Map<'hir>,
        visit_hole_span: F,
    }

    impl<'hir, F: FnMut(Span)> Visitor<'hir> for HolesVisitor<'hir, F> {
        /// - We need `NestedFilter::INTRA = true` so that `visit_item` will be called.
        /// - Bodies of nested items don't actually get visited, because of the
        ///   `visit_item` override.
        /// - For nested bodies that are not part of an item, we do want to visit any
        ///   items contained within them.
        type NestedFilter = nested_filter::All;

        fn nested_visit_map(&mut self) -> Self::Map {
            self.hir
        }

        fn visit_item(&mut self, item: &'hir hir::Item<'hir>) {
            (self.visit_hole_span)(item.span);
            // Having visited this item, we don't care about its children,
            // so don't call `walk_item`.
        }

        // We override `visit_expr` instead of the more specific expression
        // visitors, so that we have direct access to the expression span.
        fn visit_expr(&mut self, expr: &'hir hir::Expr<'hir>) {
            match expr.kind {
                hir::ExprKind::Closure(_) | hir::ExprKind::ConstBlock(_) => {
                    (self.visit_hole_span)(expr.span);
                    // Having visited this expression, we don't care about its
                    // children, so don't call `walk_expr`.
                }

                // For other expressions, recursively visit as normal.
                _ => walk_expr(self, expr),
            }
        }
    }

    let mut hole_spans = vec![];
    let mut visitor = HolesVisitor {
        hir: tcx.hir(),
        visit_hole_span: |hole_span| {
            // Discard any holes that aren't directly visible within the body span.
            if body_span.contains(hole_span) && body_span.eq_ctxt(hole_span) {
                hole_spans.push(hole_span);
            }
        },
    };

    visitor.visit_body(hir_body);
    hole_spans
}
