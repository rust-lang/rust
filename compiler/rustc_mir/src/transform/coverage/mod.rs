pub mod query;

mod debug;
mod graph;
mod spans;

use debug::{debug_options, term_type, NESTED_INDENT};
use graph::{
    BasicCoverageBlock, BasicCoverageBlockData, BcbBranch, CoverageGraph,
    TraverseCoverageGraphWithLoops,
};
use spans::{CoverageSpan, CoverageSpans};

use crate::transform::MirPass;
use crate::util::generic_graphviz::GraphvizWriter;
use crate::util::pretty;
use crate::util::spanview::{self, SpanViewable};

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::graph::dominators::Dominators;
use rustc_data_structures::graph::WithNumNodes;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::Lrc;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::{Idx, IndexVec};
use rustc_middle::hir;
use rustc_middle::hir::map::blocks::FnLikeNode;
use rustc_middle::ich::StableHashingContext;
use rustc_middle::mir::coverage::*;
use rustc_middle::mir::{
    self, BasicBlock, BasicBlockData, Coverage, SourceInfo, Statement, StatementKind, Terminator,
    TerminatorKind,
};
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;
use rustc_span::{CharPos, Pos, SourceFile, Span, Symbol};

/// A simple error message wrapper for `coverage::Error`s.
#[derive(Debug)]
pub(crate) struct Error {
    message: String,
}

impl Error {
    pub fn from_string<T>(message: String) -> Result<T, Error> {
        Err(Self { message })
    }
}

/// Inserts `StatementKind::Coverage` statements that either instrument the binary with injected
/// counters, via intrinsic `llvm.instrprof.increment`, and/or inject metadata used during codegen
/// to construct the coverage map.
pub struct InstrumentCoverage;

impl<'tcx> MirPass<'tcx> for InstrumentCoverage {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, mir_body: &mut mir::Body<'tcx>) {
        let mir_source = mir_body.source;

        // If the InstrumentCoverage pass is called on promoted MIRs, skip them.
        // See: https://github.com/rust-lang/rust/pull/73011#discussion_r438317601
        if mir_source.promoted.is_some() {
            trace!(
                "InstrumentCoverage skipped for {:?} (already promoted for Miri evaluation)",
                mir_source.def_id()
            );
            return;
        }

        let hir_id = tcx.hir().local_def_id_to_hir_id(mir_source.def_id().expect_local());
        let is_fn_like = FnLikeNode::from_node(tcx.hir().get(hir_id)).is_some();

        // Only instrument functions, methods, and closures (not constants since they are evaluated
        // at compile time by Miri).
        // FIXME(#73156): Handle source code coverage in const eval, but note, if and when const
        // expressions get coverage spans, we will probably have to "carve out" space for const
        // expressions from coverage spans in enclosing MIR's, like we do for closures. (That might
        // be tricky if const expressions have no corresponding statements in the enclosing MIR.
        // Closures are carved out by their initial `Assign` statement.)
        if !is_fn_like {
            trace!("InstrumentCoverage skipped for {:?} (not an FnLikeNode)", mir_source.def_id());
            return;
        }
        // FIXME(richkadel): By comparison, the MIR pass `ConstProp` includes associated constants,
        // with functions, methods, and closures. I assume Miri is used for associated constants as
        // well. If not, we may need to include them here too.

        trace!("InstrumentCoverage starting for {:?}", mir_source.def_id());
        Instrumentor::new(&self.name(), tcx, mir_body).inject_counters();
        trace!("InstrumentCoverage starting for {:?}", mir_source.def_id());
    }
}

struct Instrumentor<'a, 'tcx> {
    pass_name: &'a str,
    tcx: TyCtxt<'tcx>,
    mir_body: &'a mut mir::Body<'tcx>,
    hir_body: &'tcx rustc_hir::Body<'tcx>,
    bcb_dominators: Dominators<BasicCoverageBlock>,
    basic_coverage_blocks: CoverageGraph,
    function_source_hash: Option<u64>,
    next_counter_id: u32,
    num_expressions: u32,
    debug_expressions_cache: Option<FxHashMap<ExpressionOperandId, CoverageKind>>,
    debug_counters: debug::DebugCounters,
}

impl<'a, 'tcx> Instrumentor<'a, 'tcx> {
    fn new(pass_name: &'a str, tcx: TyCtxt<'tcx>, mir_body: &'a mut mir::Body<'tcx>) -> Self {
        let hir_body = hir_body(tcx, mir_body.source.def_id());
        let basic_coverage_blocks = CoverageGraph::from_mir(mir_body);
        let bcb_dominators = basic_coverage_blocks.compute_bcb_dominators();
        Self {
            pass_name,
            tcx,
            mir_body,
            hir_body,
            basic_coverage_blocks,
            bcb_dominators,
            function_source_hash: None,
            next_counter_id: CounterValueReference::START.as_u32(),
            num_expressions: 0,
            debug_expressions_cache: None,
            debug_counters: debug::DebugCounters::new(),
        }
    }

    /// Counter IDs start from one and go up.
    fn next_counter(&mut self) -> CounterValueReference {
        assert!(self.next_counter_id < u32::MAX - self.num_expressions);
        let next = self.next_counter_id;
        self.next_counter_id += 1;
        CounterValueReference::from(next)
    }

    /// Expression IDs start from u32::MAX and go down because a Expression can reference
    /// (add or subtract counts) of both Counter regions and Expression regions. The counter
    /// expression operand IDs must be unique across both types.
    fn next_expression(&mut self) -> InjectedExpressionId {
        assert!(self.next_counter_id < u32::MAX - self.num_expressions);
        let next = u32::MAX - self.num_expressions;
        self.num_expressions += 1;
        InjectedExpressionId::from(next)
    }

    fn function_source_hash(&mut self) -> u64 {
        match self.function_source_hash {
            Some(hash) => hash,
            None => {
                let hash = hash_mir_source(self.tcx, self.hir_body);
                self.function_source_hash.replace(hash);
                hash
            }
        }
    }

    fn inject_counters(&'a mut self) {
        let tcx = self.tcx;
        let source_map = tcx.sess.source_map();
        let mir_source = self.mir_body.source;
        let def_id = mir_source.def_id();
        let body_span = self.body_span();

        debug!("instrumenting {:?}, span: {}", def_id, source_map.span_to_string(body_span));

        let dump_spanview = pretty::dump_enabled(tcx, self.pass_name, def_id);
        let dump_graphviz = tcx.sess.opts.debugging_opts.dump_mir_graphviz;

        if dump_graphviz {
            self.debug_counters.enable();
        }

        let coverage_spans = CoverageSpans::generate_coverage_spans(
            &self.mir_body,
            body_span,
            &self.basic_coverage_blocks,
            &self.bcb_dominators,
        );

        // When dumping coverage spanview files, create `SpanViewables` from the `coverage_spans`.
        if dump_spanview {
            let span_viewables = self.span_viewables(&coverage_spans);
            let mut file =
                pretty::create_dump_file(tcx, "html", None, self.pass_name, &0, mir_source)
                    .expect("Unexpected error creating MIR spanview HTML file");
            let crate_name = tcx.crate_name(def_id.krate);
            let item_name = tcx.def_path(def_id).to_filename_friendly_no_crate();
            let title = format!("{}.{} - Coverage Spans", crate_name, item_name);
            spanview::write_document(tcx, def_id, span_viewables, &title, &mut file)
                .expect("Unexpected IO error dumping coverage spans as HTML");
        }

        // When debug logging, or generating the coverage graphviz output, initialize the following
        // data structures:
        let mut debug_used_expressions = debug::UsedExpressions::new();
        if level_enabled!(tracing::Level::DEBUG) || dump_graphviz {
            debug_used_expressions.enable();

            if debug_options().simplify_expressions {
                self.debug_expressions_cache.replace(FxHashMap::default());
            }
            // CAUTION! The `simplify_expressions` option is only helpful for some debugging
            // situations and it can change the generated MIR `Coverage` statements (resulting in
            // differences in behavior when enabled, under `DEBUG`, compared to normal operation and
            // testing).
            //
            // For debugging purposes, it is sometimes helpful to simplify some expression
            // equations:
            //
            //   * `x + (y - x)` becomes just `y`
            //   * `x + (y + 0)` becomes just x + y.
            //
            // Expression dependencies can deeply nested expressions, which can look quite long in
            // printed debug messages and in graphs produced by `-Zdump-graphviz`. In reality, each
            // referenced/nested expression is only present because that value is necessary to
            // compute a counter value for another part of the coverage report. Simplifying
            // expressions Does not result in less `Coverage` statements, so there is very little,
            // if any, benefit to binary size or runtime to simplifying expressions, and adds
            // additional compile-time complexity. Only enable this temporarily, if helpful to parse
            // the debug output.
        }

        // When debugging with BCB graphviz output, initialize additional data structures.
        let mut graphviz_data = debug::GraphvizData::new();
        if dump_graphviz {
            graphviz_data.enable();
        }

        // Analyze the coverage graph (aka, BCB control flow graph), and inject expression-optimized
        // counters.
        let mut collect_intermediate_expressions =
            Vec::with_capacity(self.basic_coverage_blocks.num_nodes());

        let result = self.make_bcb_counters(&coverage_spans, &mut collect_intermediate_expressions);
        if result.is_ok() {
            // If debugging, add any intermediate expressions (which are not associated with any
            // BCB) to the `debug_used_expressions` map.

            if debug_used_expressions.is_enabled() {
                for intermediate_expression in &collect_intermediate_expressions {
                    debug_used_expressions.add_expression_operands(intermediate_expression);
                }
            }

            // Inject a counter for each `CoverageSpan`.
            self.inject_coverage_span_counters(
                coverage_spans,
                &mut graphviz_data,
                &mut debug_used_expressions,
            );

            // The previous step looped through the `CoverageSpan`s and injected the counter from
            // the `CoverageSpan`s `BasicCoverageBlock`, removing it from the BCB in the process
            // (via `take_counter()`).
            //
            // Any other counter associated with a `BasicCoverageBlock`, or its incoming edge, but
            // not associated with a `CoverageSpan`, should only exist if the counter is a
            // `Expression` dependency (one of the expression operands). Collect them, and inject
            // the additional counters into the MIR, without a reportable coverage span.
            let mut bcb_counters_without_direct_coverage_spans = Vec::new();
            for (target_bcb, target_bcb_data) in self.basic_coverage_blocks.iter_enumerated_mut() {
                if let Some(counter_kind) = target_bcb_data.take_counter() {
                    bcb_counters_without_direct_coverage_spans.push((
                        None,
                        target_bcb,
                        counter_kind,
                    ));
                }
                if let Some(edge_counters) = target_bcb_data.take_edge_counters() {
                    for (from_bcb, counter_kind) in edge_counters {
                        bcb_counters_without_direct_coverage_spans.push((
                            Some(from_bcb),
                            target_bcb,
                            counter_kind,
                        ));
                    }
                }
            }

            if debug_used_expressions.is_enabled() {
                // Validate that every BCB or edge counter not directly associated with a coverage
                // span is at least indirectly associated (it is a dependency of a BCB counter that
                // _is_ associated with a coverage span).
                let mut not_validated = bcb_counters_without_direct_coverage_spans
                    .iter()
                    .map(|(_, _, counter_kind)| counter_kind)
                    .collect::<Vec<_>>();
                let mut validating_count = 0;
                while not_validated.len() != validating_count {
                    let to_validate = not_validated.split_off(0);
                    validating_count = to_validate.len();
                    for counter_kind in to_validate {
                        if debug_used_expressions.expression_is_used(counter_kind) {
                            debug_used_expressions.add_expression_operands(counter_kind);
                        } else {
                            not_validated.push(counter_kind);
                        }
                    }
                }
            }

            self.inject_indirect_counters(
                bcb_counters_without_direct_coverage_spans,
                &mut graphviz_data,
                &mut debug_used_expressions,
            );
        }

        if graphviz_data.is_enabled() {
            let node_content = |bcb| {
                self.bcb_to_string_sections(
                    self.bcb_data(bcb),
                    graphviz_data.get_bcb_coverage_spans_with_counters(bcb),
                    graphviz_data.get_bcb_dependency_counters(bcb),
                    // collect_intermediate_expressions are injected into the mir::START_BLOCK, so
                    // include them in the first BCB.
                    if bcb.index() == 0 { Some(&collect_intermediate_expressions) } else { None },
                )
            };
            let edge_labels = |from_bcb| {
                let from_terminator = self.bcb_terminator(from_bcb);
                let mut edge_labels = from_terminator.kind.fmt_successor_labels();
                edge_labels.retain(|label| label.to_string() != "unreachable");
                let edge_counters = from_terminator
                    .successors()
                    .map(|&successor_bb| graphviz_data.get_edge_counter(from_bcb, successor_bb));
                edge_labels
                    .iter()
                    .zip(edge_counters)
                    .map(|(label, some_counter)| {
                        if let Some(counter) = some_counter {
                            format!("{}\n{}", label, self.format_counter(counter))
                        } else {
                            label.to_string()
                        }
                    })
                    .collect::<Vec<_>>()
            };
            let graphviz_name = format!("Cov_{}_{}", def_id.krate.index(), def_id.index.index());
            let mut graphviz_writer = GraphvizWriter::new(
                &self.basic_coverage_blocks,
                &graphviz_name,
                node_content,
                edge_labels,
            );
            let unused_expressions = debug_used_expressions.get_unused_expressions();
            if unused_expressions.len() > 0 {
                graphviz_writer.set_graph_label(&format!(
                    "Unused expressions:\n  {}",
                    unused_expressions
                        .as_slice()
                        .iter()
                        .map(|(counter_kind, edge_from_bcb, target_bcb)| {
                            if let Some(from_bcb) = edge_from_bcb.as_ref() {
                                format!(
                                    "{:?}->{:?}: {}",
                                    from_bcb,
                                    target_bcb,
                                    self.format_counter(&counter_kind),
                                )
                            } else {
                                format!("{:?}: {}", target_bcb, self.format_counter(&counter_kind),)
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n  ")
                ));
            }
            let mut file =
                pretty::create_dump_file(tcx, "dot", None, self.pass_name, &0, mir_source)
                    .expect("Unexpected error creating BasicCoverageBlock graphviz DOT file");
            graphviz_writer
                .write_graphviz(tcx, &mut file)
                .expect("Unexpected error writing BasicCoverageBlock graphviz DOT file");
        }

        result.unwrap_or_else(|e: Error| {
            bug!("Error processing: {:?}: {:?}", self.mir_body.source.def_id(), e)
        });

        debug_used_expressions.check_no_unused(&self.debug_counters);

        for intermediate_expression in collect_intermediate_expressions {
            self.inject_intermediate_expression(intermediate_expression);
        }
    }

    /// Inject a counter for each `CoverageSpan`. There can be multiple `CoverageSpan`s for a given
    /// BCB, but only one actual counter needs to be incremented per BCB. `bb_counters` maps each
    /// `bcb` to its `Counter`, when injected. Subsequent `CoverageSpan`s for a BCB that already has
    /// a `Counter` will inject an `Expression` instead, and compute its value by adding `ZERO` to
    /// the BCB `Counter` value.
    ///
    /// If debugging, add every BCB `Expression` associated with a `CoverageSpan`s to the
    /// `used_expression_operands` map.
    fn inject_coverage_span_counters(
        &mut self,
        coverage_spans: Vec<CoverageSpan>,
        graphviz_data: &mut debug::GraphvizData,
        debug_used_expressions: &mut debug::UsedExpressions,
    ) {
        let tcx = self.tcx;
        let source_map = tcx.sess.source_map();
        let body_span = self.body_span();
        let source_file = source_map.lookup_source_file(body_span.lo());
        let file_name = Symbol::intern(&source_file.name.to_string());

        let mut bcb_counters = IndexVec::from_elem_n(None, self.basic_coverage_blocks.num_nodes());
        for covspan in coverage_spans {
            let bcb = covspan.bcb;
            let span = covspan.span;
            let counter_kind = if let Some(&counter_operand) = bcb_counters[bcb].as_ref() {
                self.make_identity_counter(counter_operand)
            } else if let Some(counter_kind) = self.bcb_data_mut(bcb).take_counter() {
                bcb_counters[bcb] = Some(counter_kind.as_operand_id());
                debug_used_expressions.add_expression_operands(&counter_kind);
                counter_kind
            } else {
                bug!("Every BasicCoverageBlock should have a Counter or Expression");
            };
            graphviz_data.add_bcb_coverage_span_with_counter(bcb, &covspan, &counter_kind);
            let some_code_region = if self.is_code_region_redundant(bcb, span, body_span) {
                None
            } else {
                Some(make_code_region(file_name, &source_file, span, body_span))
            };
            self.inject_statement(counter_kind, self.bcb_last_bb(bcb), some_code_region);
        }
    }

    fn is_code_region_redundant(
        &self,
        bcb: BasicCoverageBlock,
        span: Span,
        body_span: Span,
    ) -> bool {
        if span.hi() == body_span.hi() {
            // All functions execute a `Return`-terminated `BasicBlock`, regardless of how the
            // function returns; but only some functions also _can_ return after a `Goto` block
            // that ends on the closing brace of the function (with the `Return`). When this
            // happens, the last character is counted 2 (or possibly more) times, when we know
            // the function returned only once (of course). By giving all `Goto` terminators at
            // the end of a function a `non-reportable` code region, they are still counted
            // if appropriate, but they don't increment the line counter, as long as their is
            // also a `Return` on that last line.
            if let TerminatorKind::Goto { .. } = self.bcb_terminator(bcb).kind {
                return true;
            }
        }
        false
    }

    fn inject_indirect_counters(
        &mut self,
        bcb_counters_without_direct_coverage_spans: Vec<(
            Option<BasicCoverageBlock>,
            BasicCoverageBlock,
            CoverageKind,
        )>,
        graphviz_data: &mut debug::GraphvizData,
        debug_used_expressions: &mut debug::UsedExpressions,
    ) {
        for (edge_from_bcb, target_bcb, counter_kind) in bcb_counters_without_direct_coverage_spans
        {
            debug_used_expressions.validate_expression_is_used(
                &counter_kind,
                edge_from_bcb,
                target_bcb,
            );

            match counter_kind {
                CoverageKind::Counter { .. } => {
                    let inject_to_bb = if let Some(from_bcb) = edge_from_bcb {
                        // The MIR edge starts `from_bb` (the outgoing / last BasicBlock in
                        // `from_bcb`) and ends at `to_bb` (the incoming / first BasicBlock in the
                        // `target_bcb`; also called the `leader_bb`).
                        let from_bb = self.bcb_last_bb(from_bcb);
                        let to_bb = self.bcb_leader_bb(target_bcb);

                        debug!(
                            "Edge {:?} (last {:?}) -> {:?} (leader {:?}) requires a new MIR \
                            BasicBlock, for unclaimed edge counter {}",
                            edge_from_bcb,
                            from_bb,
                            target_bcb,
                            to_bb,
                            self.format_counter(&counter_kind),
                        );
                        debug!(
                            "  from_bb {:?} has successors: {:?}",
                            from_bb,
                            self.mir_body[from_bb].terminator().successors(),
                        );
                        let span =
                            self.mir_body[from_bb].terminator().source_info.span.shrink_to_hi();
                        let new_bb = self.mir_body.basic_blocks_mut().push(BasicBlockData {
                            statements: vec![], // counter will be injected here
                            terminator: Some(Terminator {
                                source_info: SourceInfo::outermost(span),
                                kind: TerminatorKind::Goto { target: to_bb },
                            }),
                            is_cleanup: false,
                        });
                        debug!(
                            "Edge from_bcb={:?} to to_bb={:?} has edge_counter={}",
                            from_bcb,
                            new_bb,
                            self.format_counter(&counter_kind),
                        );
                        graphviz_data.set_edge_counter(from_bcb, new_bb, &counter_kind);
                        let edge_ref = self.mir_body[from_bb]
                            .terminator_mut()
                            .successors_mut()
                            .find(|successor| **successor == to_bb)
                            .expect("from_bb should have a successor for to_bb");
                        *edge_ref = new_bb;
                        new_bb
                    } else {
                        graphviz_data.add_bcb_dependency_counter(target_bcb, &counter_kind);
                        let target_bb = self.bcb_last_bb(target_bcb);
                        debug!(
                            "{:?} ({:?}) gets a new Coverage statement for unclaimed counter {}",
                            target_bcb,
                            target_bb,
                            self.format_counter(&counter_kind),
                        );
                        target_bb
                    };

                    self.inject_statement(counter_kind, inject_to_bb, None);
                }
                CoverageKind::Expression { .. } => {
                    self.inject_intermediate_expression(counter_kind)
                }
                _ => bug!("CoverageKind should be a counter"),
            }
        }
    }

    #[inline]
    fn format_counter(&self, counter_kind: &CoverageKind) -> String {
        self.debug_counters.format_counter(counter_kind)
    }

    #[inline]
    fn bcb_leader_bb(&self, bcb: BasicCoverageBlock) -> BasicBlock {
        self.bcb_data(bcb).leader_bb()
    }

    #[inline]
    fn bcb_last_bb(&self, bcb: BasicCoverageBlock) -> BasicBlock {
        self.bcb_data(bcb).last_bb()
    }

    #[inline]
    fn bcb_terminator(&self, bcb: BasicCoverageBlock) -> &Terminator<'tcx> {
        self.bcb_data(bcb).terminator(self.mir_body)
    }

    #[inline]
    fn bcb_data(&self, bcb: BasicCoverageBlock) -> &BasicCoverageBlockData {
        &self.basic_coverage_blocks[bcb]
    }

    #[inline]
    fn bcb_data_mut(&mut self, bcb: BasicCoverageBlock) -> &mut BasicCoverageBlockData {
        &mut self.basic_coverage_blocks[bcb]
    }

    #[inline]
    fn bcb_predecessors(&self, bcb: BasicCoverageBlock) -> &Vec<BasicCoverageBlock> {
        &self.basic_coverage_blocks.predecessors[bcb]
    }

    #[inline]
    fn bcb_successors(&self, bcb: BasicCoverageBlock) -> &Vec<BasicCoverageBlock> {
        &self.basic_coverage_blocks.successors[bcb]
    }

    #[inline]
    fn bcb_branches(&self, from_bcb: BasicCoverageBlock) -> Vec<BcbBranch> {
        self.basic_coverage_blocks.successors[from_bcb]
            .iter()
            .map(|&to_bcb| BcbBranch::from_to(from_bcb, to_bcb, &self.basic_coverage_blocks))
            .collect::<Vec<_>>()
    }

    /// Returns true if the BasicCoverageBlock has zero or one incoming edge. (If zero, it should be
    /// the entry point for the function.)
    #[inline]
    fn bcb_has_one_path_to_target(&self, bcb: BasicCoverageBlock) -> bool {
        self.bcb_predecessors(bcb).len() <= 1
    }

    #[inline]
    fn bcb_is_dominated_by(&self, node: BasicCoverageBlock, dom: BasicCoverageBlock) -> bool {
        self.bcb_dominators.is_dominated_by(node, dom)
    }

    fn bcb_to_string_sections(
        &self,
        bcb_data: &BasicCoverageBlockData,
        some_coverage_spans_with_counters: Option<&Vec<(CoverageSpan, CoverageKind)>>,
        some_dependency_counters: Option<&Vec<CoverageKind>>,
        some_intermediate_expressions: Option<&Vec<CoverageKind>>,
    ) -> Vec<String> {
        let len = bcb_data.basic_blocks.len();
        let mut sections = Vec::new();
        if let Some(collect_intermediate_expressions) = some_intermediate_expressions {
            sections.push(
                collect_intermediate_expressions
                    .iter()
                    .map(|expression| format!("Intermediate {}", self.format_counter(expression)))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        }
        if let Some(coverage_spans_with_counters) = some_coverage_spans_with_counters {
            sections.push(
                coverage_spans_with_counters
                    .iter()
                    .map(|(covspan, counter)| {
                        format!(
                            "{} at {}",
                            self.format_counter(counter),
                            covspan.format(self.tcx, self.mir_body)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        }
        if let Some(dependency_counters) = some_dependency_counters {
            sections.push(format!(
                "Non-coverage counters:\n  {}",
                dependency_counters
                    .iter()
                    .map(|counter| self.format_counter(counter))
                    .collect::<Vec<_>>()
                    .join("  \n"),
            ));
        }
        if let Some(counter_kind) = &bcb_data.counter_kind {
            sections.push(format!("{:?}", counter_kind));
        }
        let non_term_blocks = bcb_data.basic_blocks[0..len - 1]
            .iter()
            .map(|&bb| format!("{:?}: {}", bb, term_type(&self.mir_body[bb].terminator().kind)))
            .collect::<Vec<_>>();
        if non_term_blocks.len() > 0 {
            sections.push(non_term_blocks.join("\n"));
        }
        sections.push(format!(
            "{:?}: {}",
            bcb_data.basic_blocks.last().unwrap(),
            term_type(&bcb_data.terminator(self.mir_body).kind)
        ));
        sections
    }

    /// Traverse the BCB CFG and add either a `Counter` or `Expression` to ever BCB, to be
    /// injected with `CoverageSpan`s. `Expressions` have no runtime overhead, so if a viable
    /// expression (adding or subtracting two other counters or expressions) can compute the same
    /// result as an embedded counter, an `Expression` should be used.
    ///
    /// If two `CoverageGraph` branch from another `BasicCoverageBlock`, one of the branches
    /// can be counted by `Expression` by subtracting the other branch from the branching
    /// block. Otherwise, the `BasicCoverageBlock` executed the least should have the `Counter`.
    /// One way to predict which branch executes the least is by considering loops. A loop is exited
    /// at a branch, so the branch that jumps to a `BasicCoverageBlock` outside the loop is almost
    /// always executed less than the branch that does not exit the loop.
    ///
    /// Returns non-code-span expressions created to represent intermediate values (if required),
    /// such as to add two counters so the result can be subtracted from another counter.
    fn make_bcb_counters(
        &mut self,
        coverage_spans: &Vec<CoverageSpan>,
        collect_intermediate_expressions: &mut Vec<CoverageKind>,
    ) -> Result<(), Error> {
        debug!("make_bcb_counters(): adding a counter or expression to each BasicCoverageBlock");
        let num_bcbs = self.basic_coverage_blocks.num_nodes();

        let mut bcbs_with_coverage = BitSet::new_empty(num_bcbs);
        for covspan in coverage_spans {
            bcbs_with_coverage.insert(covspan.bcb);
        }

        // FIXME(richkadel): Add more comments to explain the logic here and in the rest of this
        // function, and refactor this function to break it up into smaller functions that are
        // easier to understand.

        let mut traversal =
            TraverseCoverageGraphWithLoops::new(&self.basic_coverage_blocks, &self.bcb_dominators);
        while let Some(bcb) = traversal.next() {
            debug!(
                "{:?} has {} successors:",
                bcb,
                self.basic_coverage_blocks.successors[bcb].len()
            );
            for &successor in &self.basic_coverage_blocks.successors[bcb] {
                if successor == bcb {
                    debug!(
                        "{:?} has itself as its own successor. (Note, the compiled code will \
                        generate an infinite loop.)",
                        bcb
                    );
                    // Don't re-add this successor to the worklist. We are already processing it.
                    break;
                }
                for context in traversal.context_stack.iter_mut().rev() {
                    // Add successors of the current BCB to the appropriate context. Successors that
                    // stay within a loop are added to the BCBs context worklist. Successors that
                    // exit the loop (they are not dominated by the loop header) must be reachable
                    // from other BCBs outside the loop, and they will be added to a different
                    // worklist.
                    //
                    // Branching blocks (with more than one successor) must be processed before
                    // blocks with only one successor, to prevent unnecessarily complicating
                    //  Expression`s by creating a Counter in a `BasicCoverageBlock` that the
                    // branching block would have given an `Expression` (or vice versa).
                    let (some_successor_to_add, some_loop_header) =
                        if let Some((_, loop_header)) = context.loop_backedges {
                            if self.bcb_is_dominated_by(successor, loop_header) {
                                (Some(successor), Some(loop_header))
                            } else {
                                (None, None)
                            }
                        } else {
                            (Some(successor), None)
                        };
                    if let Some(successor_to_add) = some_successor_to_add {
                        if self.bcb_successors(successor_to_add).len() > 1 {
                            debug!(
                                "{:?} successor is branching. Prioritize it at the beginning of \
                                the {}",
                                successor_to_add,
                                if let Some(loop_header) = some_loop_header {
                                    format!("worklist for the loop headed by {:?}", loop_header)
                                } else {
                                    String::from("non-loop worklist")
                                },
                            );
                            context.worklist.insert(0, successor_to_add);
                        } else {
                            debug!(
                                "{:?} successor is non-branching. Defer it to the end of the {}",
                                successor_to_add,
                                if let Some(loop_header) = some_loop_header {
                                    format!("worklist for the loop headed by {:?}", loop_header)
                                } else {
                                    String::from("non-loop worklist")
                                },
                            );
                            context.worklist.push(successor_to_add);
                        }
                        break;
                    }
                }
            }

            if !bcbs_with_coverage.contains(bcb) {
                debug!(
                    "{:?} does not have any `CoverageSpan`s. A counter will only be added if \
                    and when a covered BCB has an expression dependency.",
                    bcb,
                );
                continue;
            }

            debug!("{:?} has at least one `CoverageSpan`. Get or make its counter", bcb);
            let bcb_counter_operand =
                self.get_or_make_counter_operand(bcb, collect_intermediate_expressions)?;

            let branch_needs_a_counter =
                |branch: &BcbBranch| branch.counter(&self.basic_coverage_blocks).is_none();

            let branches = self.bcb_branches(bcb);
            let needs_branch_counters =
                branches.len() > 1 && branches.iter().any(branch_needs_a_counter);

            if needs_branch_counters {
                let branching_bcb = bcb;
                let branching_counter_operand = bcb_counter_operand;

                debug!(
                    "{:?} has some branch(es) without counters:\n  {}",
                    branching_bcb,
                    branches
                        .iter()
                        .map(|branch| {
                            format!(
                                "{:?}: {:?}",
                                branch,
                                branch.counter(&self.basic_coverage_blocks)
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n  "),
                );

                // At most one of the branches (or its edge, from the branching_bcb,
                // if the branch has multiple incoming edges) can have a counter computed by
                // expression.
                //
                // If at least one of the branches leads outside of a loop (`found_loop_exit` is
                // true), and at least one other branch does not exit the loop (the first of which
                // is captured in `some_reloop_branch`), it's likely any reloop branch will be
                // executed far more often than loop exit branch, making the reloop branch a better
                // candidate for an expression.
                let mut some_reloop_branch: Option<BcbBranch> = None;
                for context in traversal.context_stack.iter().rev() {
                    if let Some((backedge_from_bcbs, _)) = &context.loop_backedges {
                        let mut found_loop_exit = false;
                        for &branch in branches.iter() {
                            if backedge_from_bcbs.iter().any(|&backedge_from_bcb| {
                                self.bcb_is_dominated_by(backedge_from_bcb, branch.target_bcb)
                            }) {
                                if let Some(reloop_branch) = some_reloop_branch {
                                    if reloop_branch.counter(&self.basic_coverage_blocks).is_none()
                                    {
                                        // we already found a candidate reloop_branch that still
                                        // needs a counter
                                        continue;
                                    }
                                }
                                // The path from branch leads back to the top of the loop. Set this
                                // branch as the `reloop_branch`. If this branch already has a
                                // counter, and we find another reloop branch that doesn't have a
                                // counter yet, that branch will be selected as the `reloop_branch`
                                // instead.
                                some_reloop_branch = Some(branch);
                            } else {
                                // The path from branch leads outside this loop
                                found_loop_exit = true;
                            }
                            if found_loop_exit
                                && some_reloop_branch.filter(branch_needs_a_counter).is_some()
                            {
                                // Found both a branch that exits the loop and a branch that returns
                                // to the top of the loop (`reloop_branch`), and the `reloop_branch`
                                // doesn't already have a counter.
                                break;
                            }
                        }
                        if !found_loop_exit {
                            debug!(
                                "No branches exit the loop, so any branch without an existing \
                                counter can have the `Expression`."
                            );
                            break;
                        }
                        if some_reloop_branch.is_some() {
                            debug!(
                                "Found a branch that exits the loop and a branch the loops back to \
                                the top of the loop (`reloop_branch`). The `reloop_branch` will \
                                get the `Expression`, as long as it still needs a counter."
                            );
                            break;
                        }
                        // else all branches exited this loop context, so run the same checks with
                        // the outer loop(s)
                    }
                }

                // Select a branch for the expression, either the recommended `reloop_branch`, or
                // if none was found, select any branch.
                let expression_branch = if let Some(reloop_branch_without_counter) =
                    some_reloop_branch.filter(branch_needs_a_counter)
                {
                    debug!(
                        "Selecting reloop_branch={:?} that still needs a counter, to get the \
                        `Expression`",
                        reloop_branch_without_counter
                    );
                    reloop_branch_without_counter
                } else {
                    let &branch_without_counter = branches
                        .iter()
                        .find(|&&branch| branch.counter(&self.basic_coverage_blocks).is_none())
                        .expect(
                            "needs_branch_counters was `true` so there should be at least one \
                            branch",
                        );
                    debug!(
                        "Selecting any branch={:?} that still needs a counter, to get the \
                        `Expression` because there was no `reloop_branch`, or it already had a \
                        counter",
                        branch_without_counter
                    );
                    branch_without_counter
                };

                // Assign a Counter or Expression to each branch, plus additional
                // `Expression`s, as needed, to sum up intermediate results.
                let mut some_sumup_counter_operand = None;
                for branch in branches {
                    if branch != expression_branch {
                        let branch_counter_operand = if branch.is_only_path_to_target() {
                            debug!(
                                "  {:?} has only one incoming edge (from {:?}), so adding a \
                                counter",
                                branch, branching_bcb
                            );
                            self.get_or_make_counter_operand(
                                branch.target_bcb,
                                collect_intermediate_expressions,
                            )?
                        } else {
                            debug!(
                                "  {:?} has multiple incoming edges, so adding an edge counter",
                                branch
                            );
                            self.get_or_make_edge_counter_operand(
                                branching_bcb,
                                branch.target_bcb,
                                collect_intermediate_expressions,
                            )?
                        };
                        if let Some(sumup_counter_operand) =
                            some_sumup_counter_operand.replace(branch_counter_operand)
                        {
                            let intermediate_expression = self.make_expression(
                                branch_counter_operand,
                                Op::Add,
                                sumup_counter_operand,
                                || None,
                            );
                            debug!(
                                "  [new intermediate expression: {}]",
                                self.format_counter(&intermediate_expression)
                            );
                            let intermediate_expression_operand =
                                intermediate_expression.as_operand_id();
                            collect_intermediate_expressions.push(intermediate_expression);
                            some_sumup_counter_operand.replace(intermediate_expression_operand);
                        }
                    }
                }
                let sumup_counter_operand =
                    some_sumup_counter_operand.expect("sumup_counter_operand should have a value");
                debug!(
                    "Making an expression for the selected expression_branch: {:?} \
                    (expression_branch predecessors: {:?})",
                    expression_branch,
                    self.bcb_predecessors(expression_branch.target_bcb),
                );
                let expression = self.make_expression(
                    branching_counter_operand,
                    Op::Subtract,
                    sumup_counter_operand,
                    || Some(format!("{:?}", expression_branch)),
                );
                debug!(
                    "{:?} gets an expression: {}",
                    expression_branch,
                    self.format_counter(&expression)
                );
                if expression_branch.is_only_path_to_target() {
                    self.bcb_data_mut(expression_branch.target_bcb).set_counter(expression)?;
                } else {
                    self.bcb_data_mut(expression_branch.target_bcb)
                        .set_edge_counter_from(branching_bcb, expression)?;
                }
            }
        }

        if traversal.is_complete() {
            Ok(())
        } else {
            Error::from_string(format!(
                "`TraverseCoverageGraphWithLoops` missed some `BasicCoverageBlock`s: {:?}",
                traversal.unvisited(),
            ))
        }
    }

    fn get_or_make_counter_operand(
        &mut self,
        bcb: BasicCoverageBlock,
        collect_intermediate_expressions: &mut Vec<CoverageKind>,
    ) -> Result<ExpressionOperandId, Error> {
        self.recursive_get_or_make_counter_operand(bcb, collect_intermediate_expressions, 1)
    }

    fn recursive_get_or_make_counter_operand(
        &mut self,
        bcb: BasicCoverageBlock,
        collect_intermediate_expressions: &mut Vec<CoverageKind>,
        debug_indent_level: usize,
    ) -> Result<ExpressionOperandId, Error> {
        Ok({
            if let Some(counter_kind) = self.basic_coverage_blocks[bcb].counter() {
                debug!(
                    "{}{:?} already has a counter: {}",
                    NESTED_INDENT.repeat(debug_indent_level),
                    bcb,
                    self.format_counter(counter_kind),
                );
                counter_kind.as_operand_id()
            } else {
                let one_path_to_target = self.bcb_has_one_path_to_target(bcb);
                if one_path_to_target || self.bcb_predecessors(bcb).contains(&bcb) {
                    let counter_kind = self.make_counter(|| Some(format!("{:?}", bcb)));
                    if one_path_to_target {
                        debug!(
                            "{}{:?} gets a new counter: {}",
                            NESTED_INDENT.repeat(debug_indent_level),
                            bcb,
                            self.format_counter(&counter_kind),
                        );
                    } else {
                        debug!(
                            "{}{:?} has itself as its own predecessor. It can't be part of its own \
                            Expression sum, so it will get its own new counter: {}. (Note, the \
                            compiled code will generate an infinite loop.)",
                            NESTED_INDENT.repeat(debug_indent_level),
                            bcb,
                            self.format_counter(&counter_kind),
                        );
                    }
                    self.basic_coverage_blocks[bcb].set_counter(counter_kind)?
                } else {
                    let mut predecessors = self.bcb_predecessors(bcb).clone().into_iter();
                    debug!(
                        "{}{:?} has multiple incoming edges and will get an expression that sums \
                        them up...",
                        NESTED_INDENT.repeat(debug_indent_level),
                        bcb,
                    );
                    let first_edge_counter_operand = self
                        .recursive_get_or_make_edge_counter_operand(
                            predecessors.next().unwrap(),
                            bcb,
                            collect_intermediate_expressions,
                            debug_indent_level + 1,
                        )?;
                    let mut some_sumup_edge_counter_operand = None;
                    for predecessor in predecessors {
                        let edge_counter_operand = self
                            .recursive_get_or_make_edge_counter_operand(
                                predecessor,
                                bcb,
                                collect_intermediate_expressions,
                                debug_indent_level + 1,
                            )?;
                        if let Some(sumup_edge_counter_operand) =
                            some_sumup_edge_counter_operand.replace(edge_counter_operand)
                        {
                            let intermediate_expression = self.make_expression(
                                sumup_edge_counter_operand,
                                Op::Add,
                                edge_counter_operand,
                                || None,
                            );
                            debug!(
                                "{}new intermediate expression: {}",
                                NESTED_INDENT.repeat(debug_indent_level),
                                self.format_counter(&intermediate_expression)
                            );
                            let intermediate_expression_operand =
                                intermediate_expression.as_operand_id();
                            collect_intermediate_expressions.push(intermediate_expression);
                            some_sumup_edge_counter_operand
                                .replace(intermediate_expression_operand);
                        }
                    }
                    let counter_kind = self.make_expression(
                        first_edge_counter_operand,
                        Op::Add,
                        some_sumup_edge_counter_operand.unwrap(),
                        || Some(format!("{:?}", bcb)),
                    );
                    debug!(
                        "{}{:?} gets a new counter (sum of predecessor counters): {}",
                        NESTED_INDENT.repeat(debug_indent_level),
                        bcb,
                        self.format_counter(&counter_kind)
                    );
                    self.basic_coverage_blocks[bcb].set_counter(counter_kind)?
                }
            }
        })
    }

    fn get_or_make_edge_counter_operand(
        &mut self,
        from_bcb: BasicCoverageBlock,
        to_bcb: BasicCoverageBlock,
        collect_intermediate_expressions: &mut Vec<CoverageKind>,
    ) -> Result<ExpressionOperandId, Error> {
        self.recursive_get_or_make_edge_counter_operand(
            from_bcb,
            to_bcb,
            collect_intermediate_expressions,
            1,
        )
    }

    fn recursive_get_or_make_edge_counter_operand(
        &mut self,
        from_bcb: BasicCoverageBlock,
        to_bcb: BasicCoverageBlock,
        collect_intermediate_expressions: &mut Vec<CoverageKind>,
        debug_indent_level: usize,
    ) -> Result<ExpressionOperandId, Error> {
        Ok({
            let successors = self.bcb_successors(from_bcb).iter();
            if successors.len() > 1 {
                if let Some(counter_kind) =
                    self.basic_coverage_blocks[to_bcb].edge_counter_from(from_bcb)
                {
                    debug!(
                        "{}Edge {:?}->{:?} already has a counter: {}",
                        NESTED_INDENT.repeat(debug_indent_level),
                        from_bcb,
                        to_bcb,
                        self.format_counter(counter_kind)
                    );
                    counter_kind.as_operand_id()
                } else {
                    let counter_kind =
                        self.make_counter(|| Some(format!("{:?}->{:?}", from_bcb, to_bcb)));
                    debug!(
                        "{}Edge {:?}->{:?} gets a new counter: {}",
                        NESTED_INDENT.repeat(debug_indent_level),
                        from_bcb,
                        to_bcb,
                        self.format_counter(&counter_kind)
                    );
                    self.basic_coverage_blocks[to_bcb]
                        .set_edge_counter_from(from_bcb, counter_kind)?
                }
            } else {
                self.recursive_get_or_make_counter_operand(
                    from_bcb,
                    collect_intermediate_expressions,
                    debug_indent_level + 1,
                )?
            }
        })
    }

    fn make_counter<F>(&mut self, block_label_fn: F) -> CoverageKind
    where
        F: Fn() -> Option<String>,
    {
        let counter = CoverageKind::Counter {
            function_source_hash: self.function_source_hash(),
            id: self.next_counter(),
        };
        if self.debug_counters.is_enabled() {
            self.debug_counters.add_counter(&counter, (block_label_fn)());
        }
        counter
    }

    fn make_expression<F>(
        &mut self,
        mut lhs: ExpressionOperandId,
        op: Op,
        mut rhs: ExpressionOperandId,
        block_label_fn: F,
    ) -> CoverageKind
    where
        F: Fn() -> Option<String>,
    {
        if let Some(expressions_cache) = self.debug_expressions_cache.as_ref() {
            if let Some(CoverageKind::Expression { lhs: lhs_lhs, op, rhs: lhs_rhs, .. }) =
                expressions_cache.get(&lhs)
            {
                if *lhs_rhs == ExpressionOperandId::ZERO {
                    lhs = *lhs_lhs;
                } else if *op == Op::Subtract && *lhs_rhs == rhs {
                    if let Some(lhs_expression) = expressions_cache.get(lhs_lhs) {
                        let expression = lhs_expression.clone();
                        return self.duplicate_expression(expression);
                    } else {
                        let counter = *lhs_lhs;
                        return self.make_identity_counter(counter);
                    }
                }
            }

            if let Some(CoverageKind::Expression { lhs: rhs_lhs, op, rhs: rhs_rhs, .. }) =
                expressions_cache.get(&rhs)
            {
                if *rhs_rhs == ExpressionOperandId::ZERO {
                    rhs = *rhs_rhs;
                } else if *op == Op::Subtract && *rhs_rhs == lhs {
                    if let Some(rhs_expression) = expressions_cache.get(rhs_lhs) {
                        let expression = rhs_expression.clone();
                        return self.duplicate_expression(expression);
                    } else {
                        let counter = *rhs_lhs;
                        return self.make_identity_counter(counter);
                    }
                }
            }
        }

        let id = self.next_expression();
        let expression = CoverageKind::Expression { id, lhs, op, rhs };
        if let Some(expressions_cache) = self.debug_expressions_cache.as_mut() {
            expressions_cache.insert(id.into(), expression.clone());
        }
        if self.debug_counters.is_enabled() {
            self.debug_counters.add_counter(&expression, (block_label_fn)());
        }
        expression
    }

    fn make_identity_counter(&mut self, counter_operand: ExpressionOperandId) -> CoverageKind {
        if let Some(expression) =
            self.debug_expressions_cache.as_ref().map_or(None, |c| c.get(&counter_operand))
        {
            let new_expression = expression.clone();
            self.duplicate_expression(new_expression)
        } else {
            let some_block_label = if self.debug_counters.is_enabled() {
                self.debug_counters.some_block_label(counter_operand).cloned()
            } else {
                None
            };
            self.make_expression(counter_operand, Op::Add, ExpressionOperandId::ZERO, || {
                some_block_label.clone()
            })
        }
    }

    fn duplicate_expression(&mut self, mut expression: CoverageKind) -> CoverageKind {
        let next_expression_id = if self.debug_expressions_cache.is_some() {
            Some(self.next_expression())
        } else {
            None
        };
        let expressions_cache = self
            .debug_expressions_cache
            .as_mut()
            .expect("`duplicate_expression()` requires the debug_expressions_cache");
        match expression {
            CoverageKind::Expression { ref mut id, .. } => {
                *id = next_expression_id.expect(
                    "next_expression_id should be Some if there is a debug_expressions_cache",
                );
                expressions_cache.insert(id.into(), expression.clone());
            }
            _ => {
                bug!("make_duplicate_expression called with non-expression type: {:?}", expression)
            }
        }
        expression
    }

    fn inject_statement(
        &mut self,
        counter_kind: CoverageKind,
        bb: BasicBlock,
        some_code_region: Option<CodeRegion>,
    ) {
        debug!(
            "  injecting statement {:?} for {:?} at code region: {:?}",
            counter_kind, bb, some_code_region
        );
        let data = &mut self.mir_body[bb];
        let source_info = data.terminator().source_info;
        let statement = Statement {
            source_info,
            kind: StatementKind::Coverage(box Coverage {
                kind: counter_kind,
                code_region: some_code_region,
            }),
        };
        data.statements.push(statement);
    }

    // Non-code expressions are injected into the coverage map, without generating executable code.
    fn inject_intermediate_expression(&mut self, expression: CoverageKind) {
        debug_assert!(if let CoverageKind::Expression { .. } = expression { true } else { false });
        debug!("  injecting non-code expression {:?}", expression);
        let inject_in_bb = mir::START_BLOCK;
        let data = &mut self.mir_body[inject_in_bb];
        let source_info = data.terminator().source_info;
        let statement = Statement {
            source_info,
            kind: StatementKind::Coverage(box Coverage { kind: expression, code_region: None }),
        };
        data.statements.push(statement);
    }

    /// Converts the computed `BasicCoverageBlockData`s into `SpanViewable`s.
    fn span_viewables(&self, coverage_spans: &Vec<CoverageSpan>) -> Vec<SpanViewable> {
        let tcx = self.tcx;
        let mut span_viewables = Vec::new();
        for coverage_span in coverage_spans {
            let tooltip = coverage_span.format_coverage_statements(tcx, self.mir_body);
            let CoverageSpan { span, bcb, .. } = coverage_span;
            let bcb_data = self.bcb_data(*bcb);
            let id = bcb_data.id();
            let leader_bb = bcb_data.leader_bb();
            span_viewables.push(SpanViewable { bb: leader_bb, span: *span, id, tooltip });
        }
        span_viewables
    }

    #[inline(always)]
    fn body_span(&self) -> Span {
        self.hir_body.value.span
    }
}

/// Convert the Span into its file name, start line and column, and end line and column
fn make_code_region(
    file_name: Symbol,
    source_file: &Lrc<SourceFile>,
    span: Span,
    body_span: Span,
) -> CodeRegion {
    let (start_line, mut start_col) = source_file.lookup_file_pos(span.lo());
    let (end_line, end_col) = if span.hi() == span.lo() {
        let (end_line, mut end_col) = (start_line, start_col);
        // Extend an empty span by one character so the region will be counted.
        let CharPos(char_pos) = start_col;
        if span.hi() == body_span.hi() {
            start_col = CharPos(char_pos - 1);
        } else {
            end_col = CharPos(char_pos + 1);
        }
        (end_line, end_col)
    } else {
        source_file.lookup_file_pos(span.hi())
    };
    CodeRegion {
        file_name,
        start_line: start_line as u32,
        start_col: start_col.to_u32() + 1,
        end_line: end_line as u32,
        end_col: end_col.to_u32() + 1,
    }
}

fn hir_body<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> &'tcx rustc_hir::Body<'tcx> {
    let hir_node = tcx.hir().get_if_local(def_id).expect("expected DefId is local");
    let fn_body_id = hir::map::associated_body(hir_node).expect("HIR node is a function with body");
    tcx.hir().body(fn_body_id)
}

fn hash_mir_source<'tcx>(tcx: TyCtxt<'tcx>, hir_body: &'tcx rustc_hir::Body<'tcx>) -> u64 {
    let mut hcx = tcx.create_no_span_stable_hashing_context();
    hash(&mut hcx, &hir_body.value).to_smaller_hash()
}

fn hash(
    hcx: &mut StableHashingContext<'tcx>,
    node: &impl HashStable<StableHashingContext<'tcx>>,
) -> Fingerprint {
    let mut stable_hasher = StableHasher::new();
    node.hash_stable(hcx, &mut stable_hasher);
    stable_hasher.finish()
}
