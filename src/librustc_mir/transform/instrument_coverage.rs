use crate::transform::{MirPass, MirSource};
use crate::util::patch::MirPatch;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir::lang_items;
use rustc_middle::hir;
use rustc_middle::ich::StableHashingContext;
use rustc_middle::mir::coverage::*;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::CoverageInfo;
use rustc_middle::mir::{
    self, traversal, BasicBlock, BasicBlockData, Operand, Place, SourceInfo, StatementKind,
    Terminator, TerminatorKind, START_BLOCK,
};
use rustc_middle::ty;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::FnDef;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;
use rustc_span::{Pos, Span};

/// Inserts call to count_code_region() as a placeholder to be replaced during code generation with
/// the intrinsic llvm.instrprof.increment.
pub struct InstrumentCoverage;

/// The `query` provider for `CoverageInfo`, requested by `codegen_intrinsic_call()` when
/// constructing the arguments for `llvm.instrprof.increment`.
pub(crate) fn provide(providers: &mut Providers) {
    providers.coverageinfo = |tcx, def_id| coverageinfo_from_mir(tcx, def_id);
}

fn coverageinfo_from_mir<'tcx>(tcx: TyCtxt<'tcx>, mir_def_id: DefId) -> CoverageInfo {
    let mir_body = tcx.optimized_mir(mir_def_id);
    // FIXME(richkadel): The current implementation assumes the MIR for the given DefId
    // represents a single function. Validate and/or correct if inlining (which should be disabled
    // if -Zinstrument-coverage is enabled) and/or monomorphization invalidates these assumptions.
    let count_code_region_fn = tcx.require_lang_item(lang_items::CountCodeRegionFnLangItem, None);
    let coverage_counter_add_fn =
        tcx.require_lang_item(lang_items::CoverageCounterAddFnLangItem, None);
    let coverage_counter_subtract_fn =
        tcx.require_lang_item(lang_items::CoverageCounterSubtractFnLangItem, None);

    // The `num_counters` argument to `llvm.instrprof.increment` is the number of injected
    // counters, with each counter having an index from `0..num_counters-1`. MIR optimization
    // may split and duplicate some BasicBlock sequences. Simply counting the calls may not
    // not work; but computing the num_counters by adding `1` to the highest index (for a given
    // instrumented function) is valid.
    //
    // `num_expressions` is the number of counter expressions added to the MIR body. Both
    // `num_counters` and `num_expressions` are used to initialize new vectors, during backend
    // code generate, to lookup counters and expressions by their simple u32 indexes.
    let mut num_counters: u32 = 0;
    let mut num_expressions: u32 = 0;
    for terminator in
        traversal::preorder(mir_body).map(|(_, data)| data).filter_map(call_terminators)
    {
        if let TerminatorKind::Call { func: Operand::Constant(func), args, .. } = &terminator.kind {
            match func.literal.ty.kind {
                FnDef(id, _) if id == count_code_region_fn => {
                    let index_arg =
                        args.get(count_code_region_args::COUNTER_INDEX).expect("arg found");
                    let counter_index = mir::Operand::scalar_from_const(index_arg)
                        .to_u32()
                        .expect("index arg is u32");
                    num_counters = std::cmp::max(num_counters, counter_index + 1);
                }
                FnDef(id, _)
                    if id == coverage_counter_add_fn || id == coverage_counter_subtract_fn =>
                {
                    let index_arg = args
                        .get(coverage_counter_expression_args::COUNTER_EXPRESSION_INDEX)
                        .expect("arg found");
                    let translated_index = mir::Operand::scalar_from_const(index_arg)
                        .to_u32()
                        .expect("index arg is u32");
                    // Counter expressions start with "translated indexes", descending from
                    // `u32::MAX`, so the range of expression indexes is disjoint from the range of
                    // counter indexes. This way, both counters and expressions can be operands in
                    // other expressions.
                    let expression_index = u32::MAX - translated_index;
                    num_expressions = std::cmp::max(num_expressions, expression_index + 1);
                }
                _ => {}
            }
        }
    }
    CoverageInfo { num_counters, num_expressions }
}

fn call_terminators(data: &'tcx BasicBlockData<'tcx>) -> Option<&'tcx Terminator<'tcx>> {
    let terminator = data.terminator();
    match terminator.kind {
        TerminatorKind::Call { .. } => Some(terminator),
        _ => None,
    }
}

impl<'tcx> MirPass<'tcx> for InstrumentCoverage {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, src: MirSource<'tcx>, mir_body: &mut mir::Body<'tcx>) {
        if tcx.sess.opts.debugging_opts.instrument_coverage {
            // If the InstrumentCoverage pass is called on promoted MIRs, skip them.
            // See: https://github.com/rust-lang/rust/pull/73011#discussion_r438317601
            if src.promoted.is_none() {
                Instrumentor::new(tcx, src, mir_body).inject_counters();
            }
        }
    }
}

/// Distinguishes the expression operators.
enum Op {
    Add,
    Subtract,
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
    fn next_counter(&mut self) -> u32 {
        assert!(self.num_counters < u32::MAX - self.num_expressions);
        let next = self.num_counters;
        self.num_counters += 1;
        next
    }

    /// Expression IDs start from u32::MAX and go down because a CounterExpression can reference
    /// (add or subtract counts) of both Counter regions and CounterExpression regions. The indexes
    /// of each type of region must be contiguous, but also must be unique across both sets.
    /// The expression IDs are eventually translated into region indexes (starting after the last
    /// counter index, for the given function), during backend code generation, by the helper method
    /// `rustc_codegen_ssa::coverageinfo::map::FunctionCoverage::translate_expressions()`.
    fn next_expression(&mut self) -> u32 {
        assert!(self.num_counters < u32::MAX - self.num_expressions);
        let next = u32::MAX - self.num_expressions;
        self.num_expressions += 1;
        next
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
        debug!(
            "instrumenting {:?}, span: {}",
            self.mir_def_id,
            self.tcx.sess.source_map().span_to_string(body_span)
        );

        // FIXME(richkadel): As a first step, counters are only injected at the top of each
        // function. The complete solution will inject counters at each conditional code branch.
        let next_block = START_BLOCK;
        self.inject_counter(body_span, next_block);

        // FIXME(richkadel): The next step to implement source based coverage analysis will be
        // instrumenting branches within functions, and some regions will be counted by "counter
        // expression". The function to inject counter expression is implemented. Replace this
        // "fake use" with real use.
        let fake_use = false;
        if fake_use {
            let add = false;
            if add {
                self.inject_counter_expression(body_span, next_block, 1, Op::Add, 2);
            } else {
                self.inject_counter_expression(body_span, next_block, 1, Op::Subtract, 2);
            }
        }
    }

    fn inject_counter(&mut self, code_region: Span, next_block: BasicBlock) -> u32 {
        let counter_id = self.next_counter();
        let function_source_hash = self.function_source_hash();
        let injection_point = code_region.shrink_to_lo();

        let count_code_region_fn = function_handle(
            self.tcx,
            self.tcx.require_lang_item(lang_items::CountCodeRegionFnLangItem, None),
            injection_point,
        );

        let mut args = Vec::new();

        use count_code_region_args::*;
        debug_assert_eq!(FUNCTION_SOURCE_HASH, args.len());
        args.push(self.const_u64(function_source_hash, injection_point));

        debug_assert_eq!(COUNTER_INDEX, args.len());
        args.push(self.const_u32(counter_id, injection_point));

        debug_assert_eq!(START_BYTE_POS, args.len());
        args.push(self.const_u32(code_region.lo().to_u32(), injection_point));

        debug_assert_eq!(END_BYTE_POS, args.len());
        args.push(self.const_u32(code_region.hi().to_u32(), injection_point));

        self.inject_call(count_code_region_fn, args, injection_point, next_block);

        counter_id
    }

    fn inject_counter_expression(
        &mut self,
        code_region: Span,
        next_block: BasicBlock,
        lhs: u32,
        op: Op,
        rhs: u32,
    ) -> u32 {
        let expression_id = self.next_expression();
        let injection_point = code_region.shrink_to_lo();

        let count_code_region_fn = function_handle(
            self.tcx,
            self.tcx.require_lang_item(
                match op {
                    Op::Add => lang_items::CoverageCounterAddFnLangItem,
                    Op::Subtract => lang_items::CoverageCounterSubtractFnLangItem,
                },
                None,
            ),
            injection_point,
        );

        let mut args = Vec::new();

        use coverage_counter_expression_args::*;
        debug_assert_eq!(COUNTER_EXPRESSION_INDEX, args.len());
        args.push(self.const_u32(expression_id, injection_point));

        debug_assert_eq!(LEFT_INDEX, args.len());
        args.push(self.const_u32(lhs, injection_point));

        debug_assert_eq!(RIGHT_INDEX, args.len());
        args.push(self.const_u32(rhs, injection_point));

        debug_assert_eq!(START_BYTE_POS, args.len());
        args.push(self.const_u32(code_region.lo().to_u32(), injection_point));

        debug_assert_eq!(END_BYTE_POS, args.len());
        args.push(self.const_u32(code_region.hi().to_u32(), injection_point));

        self.inject_call(count_code_region_fn, args, injection_point, next_block);

        expression_id
    }

    fn inject_call(
        &mut self,
        func: Operand<'tcx>,
        args: Vec<Operand<'tcx>>,
        fn_span: Span,
        next_block: BasicBlock,
    ) {
        let mut patch = MirPatch::new(self.mir_body);

        let temp = patch.new_temp(self.tcx.mk_unit(), fn_span);
        let new_block = patch.new_block(placeholder_block(fn_span));
        patch.patch_terminator(
            new_block,
            TerminatorKind::Call {
                func,
                args,
                // new_block will swapped with the next_block, after applying patch
                destination: Some((Place::from(temp), new_block)),
                cleanup: None,
                from_hir_call: false,
                fn_span,
            },
        );

        patch.add_statement(new_block.start_location(), StatementKind::StorageLive(temp));
        patch.add_statement(next_block.start_location(), StatementKind::StorageDead(temp));

        patch.apply(self.mir_body);

        // To insert the `new_block` in front of the first block in the counted branch (the
        // `next_block`), just swap the indexes, leaving the rest of the graph unchanged.
        self.mir_body.basic_blocks_mut().swap(next_block, new_block);
    }

    fn const_u32(&self, value: u32, span: Span) -> Operand<'tcx> {
        Operand::const_from_scalar(self.tcx, self.tcx.types.u32, Scalar::from_u32(value), span)
    }

    fn const_u64(&self, value: u64, span: Span) -> Operand<'tcx> {
        Operand::const_from_scalar(self.tcx, self.tcx.types.u64, Scalar::from_u64(value), span)
    }
}

fn function_handle<'tcx>(tcx: TyCtxt<'tcx>, fn_def_id: DefId, span: Span) -> Operand<'tcx> {
    let ret_ty = tcx.fn_sig(fn_def_id).output();
    let ret_ty = ret_ty.no_bound_vars().unwrap();
    let substs = tcx.mk_substs(::std::iter::once(ty::subst::GenericArg::from(ret_ty)));
    Operand::function_handle(tcx, fn_def_id, substs, span)
}

fn placeholder_block(span: Span) -> BasicBlockData<'tcx> {
    BasicBlockData {
        statements: vec![],
        terminator: Some(Terminator {
            source_info: SourceInfo::outermost(span),
            // this gets overwritten by the counter Call
            kind: TerminatorKind::Unreachable,
        }),
        is_cleanup: false,
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
