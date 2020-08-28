use crate::transform::{MirPass, MirSource};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_middle::hir;
use rustc_middle::ich::StableHashingContext;
use rustc_middle::mir;
use rustc_middle::mir::coverage::*;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{BasicBlock, Coverage, CoverageInfo, Location, Statement, StatementKind};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;
use rustc_span::{FileName, Pos, RealFileName, Span, Symbol};

/// Inserts call to count_code_region() as a placeholder to be replaced during code generation with
/// the intrinsic llvm.instrprof.increment.
pub struct InstrumentCoverage;

/// The `query` provider for `CoverageInfo`, requested by `codegen_intrinsic_call()` when
/// constructing the arguments for `llvm.instrprof.increment`.
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

fn coverageinfo_from_mir<'tcx>(tcx: TyCtxt<'tcx>, mir_def_id: DefId) -> CoverageInfo {
    let mir_body = tcx.optimized_mir(mir_def_id);

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
    fn run_pass(&self, tcx: TyCtxt<'tcx>, src: MirSource<'tcx>, mir_body: &mut mir::Body<'tcx>) {
        // If the InstrumentCoverage pass is called on promoted MIRs, skip them.
        // See: https://github.com/rust-lang/rust/pull/73011#discussion_r438317601
        if src.promoted.is_none() {
            Instrumentor::new(tcx, src, mir_body).inject_counters();
        }
    }
}

struct Instrumentor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    mir_def_id: DefId,
    mir_body: &'a mut mir::Body<'tcx>,
    hir_body: &'tcx rustc_hir::Body<'tcx>,
    function_source_hash: Option<u64>,
    num_counters: u32,
    num_expressions: u32,
}

impl<'a, 'tcx> Instrumentor<'a, 'tcx> {
    fn new(tcx: TyCtxt<'tcx>, src: MirSource<'tcx>, mir_body: &'a mut mir::Body<'tcx>) -> Self {
        let mir_def_id = src.def_id();
        let hir_body = hir_body(tcx, mir_def_id);
        Self {
            tcx,
            mir_def_id,
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
        let body_span = self.hir_body.value.span;
        debug!("instrumenting {:?}, span: {:?}", self.mir_def_id, body_span);

        // FIXME(richkadel): As a first step, counters are only injected at the top of each
        // function. The complete solution will inject counters at each conditional code branch.
        let block = rustc_middle::mir::START_BLOCK;
        let counter = self.make_counter();
        self.inject_statement(counter, body_span, block);

        // FIXME(richkadel): The next step to implement source based coverage analysis will be
        // instrumenting branches within functions, and some regions will be counted by "counter
        // expression". The function to inject counter expression is implemented. Replace this
        // "fake use" with real use.
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
    let hir_node = tcx.hir().get_if_local(def_id).expect("DefId is local");
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
