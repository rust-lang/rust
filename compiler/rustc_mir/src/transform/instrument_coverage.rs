use crate::transform::MirPass;
use crate::util::pretty;
use crate::util::spanview::{
    source_range_no_file, statement_kind_name, terminator_kind_name, write_spanview_document,
    SpanViewable, TOOLTIP_INDENT,
};

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_index::bit_set::BitSet;
use rustc_middle::hir;
use rustc_middle::ich::StableHashingContext;
use rustc_middle::mir;
use rustc_middle::mir::coverage::*;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{
    BasicBlock, BasicBlockData, Coverage, CoverageInfo, Location, Statement, StatementKind,
    TerminatorKind,
};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;
use rustc_span::{FileName, Pos, RealFileName, Span, Symbol};

/// Inserts `StatementKind::Coverage` statements that either instrument the binary with injected
/// counters, via intrinsic `llvm.instrprof.increment`, and/or inject metadata used during codegen
/// to construct the coverage map.
pub struct InstrumentCoverage;

/// The `query` provider for `CoverageInfo`, requested by `codegen_coverage()` (to inject each
/// counter) and `FunctionCoverage::new()` (to extract the coverage map metadata from the MIR).
pub(crate) fn provide(providers: &mut Providers) {
    providers.coverageinfo = |tcx, def_id| coverageinfo_from_mir(tcx, def_id);
}

struct CoverageVisitor {
    info: CoverageInfo,
}

impl Visitor<'_> for CoverageVisitor {
    fn visit_coverage(&mut self, coverage: &Coverage, _location: Location) {
        match coverage.kind {
            CoverageKind::Counter { id, .. } => {
                let counter_id = u32::from(id);
                self.info.num_counters = std::cmp::max(self.info.num_counters, counter_id + 1);
            }
            CoverageKind::Expression { id, .. } => {
                let expression_index = u32::MAX - u32::from(id);
                self.info.num_expressions =
                    std::cmp::max(self.info.num_expressions, expression_index + 1);
            }
            _ => {}
        }
    }
}

fn coverageinfo_from_mir<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> CoverageInfo {
    let mir_body = tcx.optimized_mir(def_id);

    // The `num_counters` argument to `llvm.instrprof.increment` is the number of injected
    // counters, with each counter having a counter ID from `0..num_counters-1`. MIR optimization
    // may split and duplicate some BasicBlock sequences. Simply counting the calls may not
    // work; but computing the num_counters by adding `1` to the highest counter_id (for a given
    // instrumented function) is valid.
    //
    // `num_expressions` is the number of counter expressions added to the MIR body. Both
    // `num_counters` and `num_expressions` are used to initialize new vectors, during backend
    // code generate, to lookup counters and expressions by simple u32 indexes.
    let mut coverage_visitor =
        CoverageVisitor { info: CoverageInfo { num_counters: 0, num_expressions: 0 } };

    coverage_visitor.visit_body(mir_body);
    coverage_visitor.info
}

impl<'tcx> MirPass<'tcx> for InstrumentCoverage {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, mir_body: &mut mir::Body<'tcx>) {
        // If the InstrumentCoverage pass is called on promoted MIRs, skip them.
        // See: https://github.com/rust-lang/rust/pull/73011#discussion_r438317601
        if mir_body.source.promoted.is_none() {
            Instrumentor::new(&self.name(), tcx, mir_body).inject_counters();
        }
    }
}

#[derive(Clone)]
struct CoverageRegion {
    pub span: Span,
    pub blocks: Vec<BasicBlock>,
}

struct Instrumentor<'a, 'tcx> {
    pass_name: &'a str,
    tcx: TyCtxt<'tcx>,
    mir_body: &'a mut mir::Body<'tcx>,
    hir_body: &'tcx rustc_hir::Body<'tcx>,
    function_source_hash: Option<u64>,
    num_counters: u32,
    num_expressions: u32,
}

impl<'a, 'tcx> Instrumentor<'a, 'tcx> {
    fn new(pass_name: &'a str, tcx: TyCtxt<'tcx>, mir_body: &'a mut mir::Body<'tcx>) -> Self {
        let hir_body = hir_body(tcx, mir_body.source.def_id());
        Self {
            pass_name,
            tcx,
            mir_body,
            hir_body,
            function_source_hash: None,
            num_counters: 0,
            num_expressions: 0,
        }
    }

    /// Counter IDs start from zero and go up.
    fn next_counter(&mut self) -> CounterValueReference {
        assert!(self.num_counters < u32::MAX - self.num_expressions);
        let next = self.num_counters;
        self.num_counters += 1;
        CounterValueReference::from(next)
    }

    /// Expression IDs start from u32::MAX and go down because a CounterExpression can reference
    /// (add or subtract counts) of both Counter regions and CounterExpression regions. The counter
    /// expression operand IDs must be unique across both types.
    fn next_expression(&mut self) -> InjectedExpressionIndex {
        assert!(self.num_counters < u32::MAX - self.num_expressions);
        let next = u32::MAX - self.num_expressions;
        self.num_expressions += 1;
        InjectedExpressionIndex::from(next)
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

    fn inject_counters(&mut self) {
        let tcx = self.tcx;
        let def_id = self.mir_body.source.def_id();
        let mir_body = &self.mir_body;
        let body_span = self.hir_body.value.span;
        debug!(
            "instrumenting {:?}, span: {}",
            def_id,
            tcx.sess.source_map().span_to_string(body_span)
        );

        if !tcx.sess.opts.debugging_opts.experimental_coverage {
            // Coverage at the function level should be accurate. This is the default implementation
            // if `-Z experimental-coverage` is *NOT* enabled.
            let block = rustc_middle::mir::START_BLOCK;
            let counter = self.make_counter();
            self.inject_statement(counter, body_span, block);
            return;
        }
        // FIXME(richkadel): else if `-Z experimental-coverage` *IS* enabled: Efforts are still in
        // progress to identify the correct code region spans and associated counters to generate
        // accurate Rust coverage reports.

        let block_span = |data: &BasicBlockData<'tcx>| {
            // The default span will be the `Terminator` span; but until we have a smarter solution,
            // the coverage region also incorporates at least the statements in this BasicBlock as
            // well. Extend the span to encompass all, if possible.
            // FIXME(richkadel): Assuming the terminator's span is already known to be contained in `body_span`.
            let mut span = data.terminator().source_info.span;
            // FIXME(richkadel): It's looking unlikely that we should compute a span from MIR
            // spans, but if we do keep something like this logic, we will need a smarter way
            // to combine `Statement`s and/or `Terminator`s with `Span`s from different
            // files.
            for statement_span in data.statements.iter().map(|statement| statement.source_info.span)
            {
                // Only combine Spans from the function's body_span.
                if body_span.contains(statement_span) {
                    span = span.to(statement_span);
                }
            }
            span
        };

        // Traverse the CFG but ignore anything following an `unwind`
        let cfg_without_unwind = ShortCircuitPreorder::new(mir_body, |term_kind| {
            let mut successors = term_kind.successors();
            match &term_kind {
                // SwitchInt successors are never unwind, and all of them should be traversed
                TerminatorKind::SwitchInt { .. } => successors,
                // For all other kinds, return only the first successor, if any, and ignore unwinds
                _ => successors.next().into_iter().chain(&[]),
            }
        });

        let mut coverage_regions = Vec::with_capacity(cfg_without_unwind.size_hint().0);
        for (bb, data) in cfg_without_unwind {
            if !body_span.contains(data.terminator().source_info.span) {
                continue;
            }

            // FIXME(richkadel): Regions will soon contain multiple blocks.
            let mut blocks = Vec::new();
            blocks.push(bb);
            let span = block_span(data);
            coverage_regions.push(CoverageRegion { span, blocks });
        }

        let span_viewables = if pretty::dump_enabled(tcx, self.pass_name, def_id) {
            Some(self.span_viewables(&coverage_regions))
        } else {
            None
        };

        // Inject counters for the selected spans
        for CoverageRegion { span, blocks } in coverage_regions {
            debug!(
                "Injecting counter at: {:?}:\n{}\n==========",
                span,
                tcx.sess.source_map().span_to_snippet(span).expect("Error getting source for span"),
            );
            let counter = self.make_counter();
            self.inject_statement(counter, span, blocks[0]);
        }

        if let Some(span_viewables) = span_viewables {
            let mut file = pretty::create_dump_file(
                tcx,
                "html",
                None,
                self.pass_name,
                &0,
                self.mir_body.source,
            )
            .expect("Unexpected error creating MIR spanview HTML file");
            write_spanview_document(tcx, def_id, span_viewables, &mut file)
                .expect("Unexpected IO error dumping coverage spans as HTML");
        }

        // FIXME(richkadel): Some regions will be counted by "counter expression". Counter
        // expressions are supported, but are not yet generated. When they are, remove this `fake_use`
        // block.
        let fake_use = false;
        if fake_use {
            let add = false;
            let fake_counter = CoverageKind::Counter {
                function_source_hash: self.function_source_hash(),
                id: CounterValueReference::from_u32(1),
            };
            let fake_expression = CoverageKind::Expression {
                id: InjectedExpressionIndex::from(u32::MAX - 1),
                lhs: ExpressionOperandId::from_u32(1),
                op: Op::Add,
                rhs: ExpressionOperandId::from_u32(2),
            };

            let lhs = fake_counter.as_operand_id();
            let op = if add { Op::Add } else { Op::Subtract };
            let rhs = fake_expression.as_operand_id();

            let block = rustc_middle::mir::START_BLOCK;

            let expression = self.make_expression(lhs, op, rhs);
            self.inject_statement(expression, body_span, block);
        }
    }

    fn make_counter(&mut self) -> CoverageKind {
        CoverageKind::Counter {
            function_source_hash: self.function_source_hash(),
            id: self.next_counter(),
        }
    }

    fn make_expression(
        &mut self,
        lhs: ExpressionOperandId,
        op: Op,
        rhs: ExpressionOperandId,
    ) -> CoverageKind {
        CoverageKind::Expression { id: self.next_expression(), lhs, op, rhs }
    }

    fn inject_statement(&mut self, coverage_kind: CoverageKind, span: Span, block: BasicBlock) {
        let code_region = make_code_region(self.tcx, &span);
        debug!("  injecting statement {:?} covering {:?}", coverage_kind, code_region);

        let data = &mut self.mir_body[block];
        let source_info = data.terminator().source_info;
        let statement = Statement {
            source_info,
            kind: StatementKind::Coverage(box Coverage { kind: coverage_kind, code_region }),
        };
        data.statements.push(statement);
    }

    /// Converts the computed `CoverageRegion`s into `SpanViewable`s.
    fn span_viewables(&self, coverage_regions: &Vec<CoverageRegion>) -> Vec<SpanViewable> {
        let mut span_viewables = Vec::new();
        for coverage_region in coverage_regions {
            span_viewables.push(SpanViewable {
                span: coverage_region.span,
                id: format!("{}", coverage_region.blocks[0].index()),
                tooltip: self.make_tooltip_text(coverage_region),
            });
        }
        span_viewables
    }

    /// A custom tooltip renderer used in a spanview HTML+CSS document used for coverage analysis.
    fn make_tooltip_text(&self, coverage_region: &CoverageRegion) -> String {
        const INCLUDE_COVERAGE_STATEMENTS: bool = false;
        let tcx = self.tcx;
        let source_map = tcx.sess.source_map();
        let mut text = Vec::new();
        for (i, &bb) in coverage_region.blocks.iter().enumerate() {
            if i > 0 {
                text.push("\n".to_owned());
            }
            text.push(format!("{:?}: {}:", bb, &source_map.span_to_string(coverage_region.span)));
            let data = &self.mir_body.basic_blocks()[bb];
            for statement in &data.statements {
                let statement_string = match statement.kind {
                    StatementKind::Coverage(box ref coverage) => match coverage.kind {
                        CoverageKind::Counter { id, .. } => {
                            if !INCLUDE_COVERAGE_STATEMENTS {
                                continue;
                            }
                            format!("increment counter #{}", id.index())
                        }
                        CoverageKind::Expression { id, lhs, op, rhs } => {
                            if !INCLUDE_COVERAGE_STATEMENTS {
                                continue;
                            }
                            format!(
                                "expression #{} = {} {} {}",
                                id.index(),
                                lhs.index(),
                                if op == Op::Add { "+" } else { "-" },
                                rhs.index()
                            )
                        }
                        CoverageKind::Unreachable => {
                            if !INCLUDE_COVERAGE_STATEMENTS {
                                continue;
                            }
                            String::from("unreachable")
                        }
                    },
                    _ => format!("{:?}", statement),
                };
                let source_range = source_range_no_file(tcx, &statement.source_info.span);
                text.push(format!(
                    "\n{}{}: {}: {}",
                    TOOLTIP_INDENT,
                    source_range,
                    statement_kind_name(statement),
                    statement_string
                ));
            }
            let term = data.terminator();
            let source_range = source_range_no_file(tcx, &term.source_info.span);
            text.push(format!(
                "\n{}{}: {}: {:?}",
                TOOLTIP_INDENT,
                source_range,
                terminator_kind_name(term),
                term.kind
            ));
        }
        text.join("")
    }
}

/// Convert the Span into its file name, start line and column, and end line and column
fn make_code_region<'tcx>(tcx: TyCtxt<'tcx>, span: &Span) -> CodeRegion {
    let source_map = tcx.sess.source_map();
    let start = source_map.lookup_char_pos(span.lo());
    let end = if span.hi() == span.lo() {
        start.clone()
    } else {
        let end = source_map.lookup_char_pos(span.hi());
        debug_assert_eq!(
            start.file.name,
            end.file.name,
            "Region start ({:?} -> {:?}) and end ({:?} -> {:?}) don't come from the same source file!",
            span.lo(),
            start,
            span.hi(),
            end
        );
        end
    };
    match &start.file.name {
        FileName::Real(RealFileName::Named(path)) => CodeRegion {
            file_name: Symbol::intern(&path.to_string_lossy()),
            start_line: start.line as u32,
            start_col: start.col.to_u32() + 1,
            end_line: end.line as u32,
            end_col: end.col.to_u32() + 1,
        },
        _ => bug!("start.file.name should be a RealFileName, but it was: {:?}", start.file.name),
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

pub struct ShortCircuitPreorder<
    'a,
    'tcx,
    F: Fn(&'tcx TerminatorKind<'tcx>) -> mir::Successors<'tcx>,
> {
    body: &'a mir::Body<'tcx>,
    visited: BitSet<BasicBlock>,
    worklist: Vec<BasicBlock>,
    filtered_successors: F,
}

impl<'a, 'tcx, F: Fn(&'tcx TerminatorKind<'tcx>) -> mir::Successors<'tcx>>
    ShortCircuitPreorder<'a, 'tcx, F>
{
    pub fn new(
        body: &'a mir::Body<'tcx>,
        filtered_successors: F,
    ) -> ShortCircuitPreorder<'a, 'tcx, F> {
        let worklist = vec![mir::START_BLOCK];

        ShortCircuitPreorder {
            body,
            visited: BitSet::new_empty(body.basic_blocks().len()),
            worklist,
            filtered_successors,
        }
    }
}

impl<'a: 'tcx, 'tcx, F: Fn(&'tcx TerminatorKind<'tcx>) -> mir::Successors<'tcx>> Iterator
    for ShortCircuitPreorder<'a, 'tcx, F>
{
    type Item = (BasicBlock, &'a BasicBlockData<'tcx>);

    fn next(&mut self) -> Option<(BasicBlock, &'a BasicBlockData<'tcx>)> {
        while let Some(idx) = self.worklist.pop() {
            if !self.visited.insert(idx) {
                continue;
            }

            let data = &self.body[idx];

            if let Some(ref term) = data.terminator {
                self.worklist.extend((self.filtered_successors)(&term.kind));
            }

            return Some((idx, data));
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.body.basic_blocks().len() - self.visited.count();
        (size, Some(size))
    }
}
