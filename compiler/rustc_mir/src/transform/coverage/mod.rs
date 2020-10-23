pub mod query;

mod counters;
mod debug;
mod graph;
mod spans;

use counters::CoverageCounters;
use graph::CoverageGraph;
use spans::{CoverageSpan, CoverageSpans};

use crate::transform::MirPass;
use crate::util::pretty;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::graph::WithNumNodes;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::Lrc;
use rustc_index::vec::IndexVec;
use rustc_middle::hir;
use rustc_middle::hir::map::blocks::FnLikeNode;
use rustc_middle::ich::StableHashingContext;
use rustc_middle::mir::coverage::*;
use rustc_middle::mir::{self, BasicBlock, Coverage, Statement, StatementKind};
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;
use rustc_span::{CharPos, Pos, SourceFile, Span, Symbol};

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
    body_span: Span,
    basic_coverage_blocks: CoverageGraph,
    coverage_counters: CoverageCounters,
}

impl<'a, 'tcx> Instrumentor<'a, 'tcx> {
    fn new(pass_name: &'a str, tcx: TyCtxt<'tcx>, mir_body: &'a mut mir::Body<'tcx>) -> Self {
        let hir_body = hir_body(tcx, mir_body.source.def_id());
        let body_span = hir_body.value.span;
        let function_source_hash = hash_mir_source(tcx, hir_body);
        let basic_coverage_blocks = CoverageGraph::from_mir(mir_body);
        Self {
            pass_name,
            tcx,
            mir_body,
            body_span,
            basic_coverage_blocks,
            coverage_counters: CoverageCounters::new(function_source_hash),
        }
    }

    fn inject_counters(&'a mut self) {
        let tcx = self.tcx;
        let source_map = tcx.sess.source_map();
        let mir_source = self.mir_body.source;
        let def_id = mir_source.def_id();
        let body_span = self.body_span;

        debug!("instrumenting {:?}, span: {}", def_id, source_map.span_to_string(body_span));

        let mut graphviz_data = debug::GraphvizData::new();

        let dump_graphviz = tcx.sess.opts.debugging_opts.dump_mir_graphviz;
        if dump_graphviz {
            graphviz_data.enable();
            self.coverage_counters.enable_debug();
        }

        ////////////////////////////////////////////////////
        // Compute `CoverageSpan`s from the `CoverageGraph`.
        let coverage_spans = CoverageSpans::generate_coverage_spans(
            &self.mir_body,
            body_span,
            &self.basic_coverage_blocks,
        );

        if pretty::dump_enabled(tcx, self.pass_name, def_id) {
            debug::dump_coverage_spanview(
                tcx,
                self.mir_body,
                &self.basic_coverage_blocks,
                self.pass_name,
                &coverage_spans,
            );
        }

        self.inject_coverage_span_counters(coverage_spans, &mut graphviz_data);

        if graphviz_data.is_enabled() {
            // Even if there was an error, a partial CoverageGraph can still generate a useful
            // graphviz output.
            debug::dump_coverage_graphviz(
                tcx,
                self.mir_body,
                self.pass_name,
                &self.basic_coverage_blocks,
                &self.coverage_counters.debug_counters,
                &graphviz_data,
            );
        }
    }

    /// Inject a counter for each `CoverageSpan`. There can be multiple `CoverageSpan`s for a given
    /// BCB, but only one actual counter needs to be incremented per BCB. `bcb_counters` maps each
    /// `bcb` to its `Counter`, when injected. Subsequent `CoverageSpan`s for a BCB that already has
    /// a `Counter` will inject an `Expression` instead, and compute its value by adding `ZERO` to
    /// the BCB `Counter` value.
    fn inject_coverage_span_counters(
        &mut self,
        coverage_spans: Vec<CoverageSpan>,
        graphviz_data: &mut debug::GraphvizData,
    ) {
        let tcx = self.tcx;
        let source_map = tcx.sess.source_map();
        let body_span = self.body_span;
        let source_file = source_map.lookup_source_file(body_span.lo());
        let file_name = Symbol::intern(&source_file.name.to_string());

        let mut bcb_counters = IndexVec::from_elem_n(None, self.basic_coverage_blocks.num_nodes());
        for covspan in coverage_spans {
            let bcb = covspan.bcb;
            let span = covspan.span;
            if let Some(&counter_operand) = bcb_counters[bcb].as_ref() {
                let expression = self.coverage_counters.make_expression(
                    counter_operand,
                    Op::Add,
                    ExpressionOperandId::ZERO,
                    || Some(format!("{:?}", bcb)),
                );
                debug!(
                    "Injecting counter expression {:?} at: {:?}:\n{}\n==========",
                    expression,
                    span,
                    source_map.span_to_snippet(span).expect("Error getting source for span"),
                );
                graphviz_data.add_bcb_coverage_span_with_counter(bcb, &covspan, &expression);
                let bb = self.basic_coverage_blocks[bcb].leader_bb();
                let code_region = make_code_region(file_name, &source_file, span, body_span);
                inject_statement(self.mir_body, expression, bb, Some(code_region));
            } else {
                let counter = self.coverage_counters.make_counter(|| Some(format!("{:?}", bcb)));
                debug!(
                    "Injecting counter {:?} at: {:?}:\n{}\n==========",
                    counter,
                    span,
                    source_map.span_to_snippet(span).expect("Error getting source for span"),
                );
                let counter_operand = counter.as_operand_id();
                bcb_counters[bcb] = Some(counter_operand);
                graphviz_data.add_bcb_coverage_span_with_counter(bcb, &covspan, &counter);
                let bb = self.basic_coverage_blocks[bcb].leader_bb();
                let code_region = make_code_region(file_name, &source_file, span, body_span);
                inject_statement(self.mir_body, counter, bb, Some(code_region));
            }
        }
    }
}

fn inject_statement(
    mir_body: &mut mir::Body<'tcx>,
    counter_kind: CoverageKind,
    bb: BasicBlock,
    some_code_region: Option<CodeRegion>,
) {
    debug!(
        "  injecting statement {:?} for {:?} at code region: {:?}",
        counter_kind, bb, some_code_region
    );
    let data = &mut mir_body[bb];
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
