use crate::transform::{MirPass, MirSource};
use crate::util::patch::MirPatch;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir::lang_items;
use rustc_middle::hir;
use rustc_middle::ich::StableHashingContext;
use rustc_middle::mir::coverage::*;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::{
    self, traversal, BasicBlock, BasicBlockData, CoverageInfo, Operand, Place, SourceInfo,
    SourceScope, StatementKind, Terminator, TerminatorKind,
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
    // counters, with each counter having a counter ID from `0..num_counters-1`. MIR optimization
    // may split and duplicate some BasicBlock sequences. Simply counting the calls may not
    // work; but computing the num_counters by adding `1` to the highest counter_id (for a given
    // instrumented function) is valid.
    //
    // `num_expressions` is the number of counter expressions added to the MIR body. Both
    // `num_counters` and `num_expressions` are used to initialize new vectors, during backend
    // code generate, to lookup counters and expressions by simple u32 indexes.
    let mut num_counters: u32 = 0;
    let mut num_expressions: u32 = 0;
    for terminator in
        traversal::preorder(mir_body).map(|(_, data)| data).filter_map(call_terminators)
    {
        if let TerminatorKind::Call { func: Operand::Constant(func), args, .. } = &terminator.kind {
            match func.literal.ty.kind {
                FnDef(id, _) if id == count_code_region_fn => {
                    let counter_id_arg =
                        args.get(count_code_region_args::COUNTER_ID).expect("arg found");
                    let counter_id = mir::Operand::scalar_from_const(counter_id_arg)
                        .to_u32()
                        .expect("counter_id arg is u32");
                    num_counters = std::cmp::max(num_counters, counter_id + 1);
                }
                FnDef(id, _)
                    if id == coverage_counter_add_fn || id == coverage_counter_subtract_fn =>
                {
                    let expression_id_arg = args
                        .get(coverage_counter_expression_args::EXPRESSION_ID)
                        .expect("arg found");
                    let id_descending_from_max = mir::Operand::scalar_from_const(expression_id_arg)
                        .to_u32()
                        .expect("expression_id arg is u32");
                    // Counter expressions are initially assigned IDs descending from `u32::MAX`, so
                    // the range of expression IDs is disjoint from the range of counter IDs. This
                    // way, both counters and expressions can be operands in other expressions.
                    let expression_index = u32::MAX - id_descending_from_max;
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
        // If the InstrumentCoverage pass is called on promoted MIRs, skip them.
        // See: https://github.com/rust-lang/rust/pull/73011#discussion_r438317601
        if src.promoted.is_none() {
            Instrumentor::new(tcx, src, mir_body).inject_counters();
        }
    }
}

/// Distinguishes the expression operators.
enum Op {
    Add,
    Subtract,
}

struct InjectedCall<'tcx> {
    func: Operand<'tcx>,
    args: Vec<Operand<'tcx>>,
    inject_at: Span,
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
    /// (add or subtract counts) of both Counter regions and CounterExpression regions. The counter
    /// expression operand IDs must be unique across both types.
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
        let mir_body = &self.mir_body;
        let body_span = self.hir_body.value.span;
        debug!("instrumenting {:?}, span: {:?}", self.mir_def_id, body_span);

        // FIXME(richkadel): As a first step, counters are only injected at the top of each
        // function. The complete solution will inject counters at each conditional code branch.
        let _ignore = mir_body;
        let id = self.next_counter();
        let function_source_hash = self.function_source_hash();
        let code_region = body_span;
        let scope = rustc_middle::mir::OUTERMOST_SOURCE_SCOPE;
        let is_cleanup = false;
        let next_block = rustc_middle::mir::START_BLOCK;
        self.inject_call(
            self.make_counter(id, function_source_hash, code_region),
            scope,
            is_cleanup,
            next_block,
        );

        // FIXME(richkadel): The next step to implement source based coverage analysis will be
        // instrumenting branches within functions, and some regions will be counted by "counter
        // expression". The function to inject counter expression is implemented. Replace this
        // "fake use" with real use.
        let fake_use = false;
        if fake_use {
            let add = false;
            let lhs = 1;
            let op = if add { Op::Add } else { Op::Subtract };
            let rhs = 2;

            let code_region = body_span;
            let scope = rustc_middle::mir::OUTERMOST_SOURCE_SCOPE;
            let is_cleanup = false;
            let next_block = rustc_middle::mir::START_BLOCK;

            let id = self.next_expression();
            self.inject_call(
                self.make_expression(id, code_region, lhs, op, rhs),
                scope,
                is_cleanup,
                next_block,
            );
        }
    }

    fn make_counter(
        &self,
        id: u32,
        function_source_hash: u64,
        code_region: Span,
    ) -> InjectedCall<'tcx> {
        let inject_at = code_region.shrink_to_lo();

        let func = function_handle(
            self.tcx,
            self.tcx.require_lang_item(lang_items::CountCodeRegionFnLangItem, None),
            inject_at,
        );

        let mut args = Vec::new();

        use count_code_region_args::*;
        debug_assert_eq!(FUNCTION_SOURCE_HASH, args.len());
        args.push(self.const_u64(function_source_hash, inject_at));

        debug_assert_eq!(COUNTER_ID, args.len());
        args.push(self.const_u32(id, inject_at));

        debug_assert_eq!(START_BYTE_POS, args.len());
        args.push(self.const_u32(code_region.lo().to_u32(), inject_at));

        debug_assert_eq!(END_BYTE_POS, args.len());
        args.push(self.const_u32(code_region.hi().to_u32(), inject_at));

        InjectedCall { func, args, inject_at }
    }

    fn make_expression(
        &self,
        id: u32,
        code_region: Span,
        lhs: u32,
        op: Op,
        rhs: u32,
    ) -> InjectedCall<'tcx> {
        let inject_at = code_region.shrink_to_lo();

        let func = function_handle(
            self.tcx,
            self.tcx.require_lang_item(
                match op {
                    Op::Add => lang_items::CoverageCounterAddFnLangItem,
                    Op::Subtract => lang_items::CoverageCounterSubtractFnLangItem,
                },
                None,
            ),
            inject_at,
        );

        let mut args = Vec::new();

        use coverage_counter_expression_args::*;
        debug_assert_eq!(EXPRESSION_ID, args.len());
        args.push(self.const_u32(id, inject_at));

        debug_assert_eq!(LEFT_ID, args.len());
        args.push(self.const_u32(lhs, inject_at));

        debug_assert_eq!(RIGHT_ID, args.len());
        args.push(self.const_u32(rhs, inject_at));

        debug_assert_eq!(START_BYTE_POS, args.len());
        args.push(self.const_u32(code_region.lo().to_u32(), inject_at));

        debug_assert_eq!(END_BYTE_POS, args.len());
        args.push(self.const_u32(code_region.hi().to_u32(), inject_at));

        InjectedCall { func, args, inject_at }
    }

    fn inject_call(
        &mut self,
        call: InjectedCall<'tcx>,
        scope: SourceScope,
        is_cleanup: bool,
        next_block: BasicBlock,
    ) {
        let InjectedCall { func, args, inject_at } = call;
        debug!(
            "  injecting {}call to {:?}({:?}) at: {:?}, scope: {:?}",
            if is_cleanup { "cleanup " } else { "" },
            func,
            args,
            inject_at,
            scope,
        );

        let mut patch = MirPatch::new(self.mir_body);

        let temp = patch.new_temp(self.tcx.mk_unit(), inject_at);
        let new_block = patch.new_block(placeholder_block(inject_at, scope, is_cleanup));
        patch.patch_terminator(
            new_block,
            TerminatorKind::Call {
                func,
                args,
                // new_block will swapped with the next_block, after applying patch
                destination: Some((Place::from(temp), new_block)),
                cleanup: None,
                from_hir_call: false,
                fn_span: inject_at,
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

fn placeholder_block(span: Span, scope: SourceScope, is_cleanup: bool) -> BasicBlockData<'tcx> {
    BasicBlockData {
        statements: vec![],
        terminator: Some(Terminator {
            source_info: SourceInfo { span, scope },
            // this gets overwritten by the counter Call
            kind: TerminatorKind::Unreachable,
        }),
        is_cleanup,
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
