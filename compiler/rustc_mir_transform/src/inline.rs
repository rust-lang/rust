//! Inlining pass for MIR functions.

use std::assert_matches::debug_assert_matches;
use std::iter;
use std::ops::{Range, RangeFrom};

use rustc_abi::{ExternAbi, FieldIdx};
use rustc_hir::attrs::{InlineAttr, OptimizeAttr};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_index::Idx;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::bug;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Instance, InstanceKind, Ty, TyCtxt, TypeFlags, TypeVisitableExt};
use rustc_session::config::{DebugInfo, OptLevel};
use rustc_span::source_map::Spanned;
use tracing::{debug, instrument, trace, trace_span};

use crate::cost_checker::{CostChecker, is_call_like};
use crate::deref_separator::deref_finder;
use crate::simplify::{UsedInStmtLocals, simplify_cfg};
use crate::validate::validate_types;
use crate::{check_inline, util};

pub(crate) mod cycle;

const HISTORY_DEPTH_LIMIT: usize = 20;
const TOP_DOWN_DEPTH_LIMIT: usize = 5;

#[derive(Clone, Debug)]
struct CallSite<'tcx> {
    callee: Instance<'tcx>,
    fn_sig: ty::PolyFnSig<'tcx>,
    block: BasicBlock,
    source_info: SourceInfo,
}

// Made public so that `mir_drops_elaborated_and_const_checked` can be overridden
// by custom rustc drivers, running all the steps by themselves. See #114628.
pub struct Inline;

impl<'tcx> crate::MirPass<'tcx> for Inline {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        if let Some(enabled) = sess.opts.unstable_opts.inline_mir {
            return enabled;
        }

        match sess.mir_opt_level() {
            0 | 1 => false,
            2 => {
                (sess.opts.optimize == OptLevel::More || sess.opts.optimize == OptLevel::Aggressive)
                    && sess.opts.incremental == None
            }
            _ => true,
        }
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let span = trace_span!("inline", body = %tcx.def_path_str(body.source.def_id()));
        let _guard = span.enter();
        if inline::<NormalInliner<'tcx>>(tcx, body) {
            debug!("running simplify cfg on {:?}", body.source);
            simplify_cfg(tcx, body);
            deref_finder(tcx, body);
        }
    }

    fn is_required(&self) -> bool {
        false
    }
}

pub struct ForceInline;

impl ForceInline {
    pub fn should_run_pass_for_callee<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> bool {
        matches!(tcx.codegen_fn_attrs(def_id).inline, InlineAttr::Force { .. })
    }
}

impl<'tcx> crate::MirPass<'tcx> for ForceInline {
    fn is_enabled(&self, _: &rustc_session::Session) -> bool {
        true
    }

    fn can_be_overridden(&self) -> bool {
        false
    }

    fn is_required(&self) -> bool {
        true
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let span = trace_span!("force_inline", body = %tcx.def_path_str(body.source.def_id()));
        let _guard = span.enter();
        if inline::<ForceInliner<'tcx>>(tcx, body) {
            debug!("running simplify cfg on {:?}", body.source);
            simplify_cfg(tcx, body);
            deref_finder(tcx, body);
        }
    }
}

trait Inliner<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, def_id: DefId, body: &Body<'tcx>) -> Self;

    fn tcx(&self) -> TyCtxt<'tcx>;
    fn typing_env(&self) -> ty::TypingEnv<'tcx>;
    fn history(&self) -> &[DefId];
    fn caller_def_id(&self) -> DefId;

    /// Has the caller body been changed?
    fn changed(self) -> bool;

    /// Should inlining happen for a given callee?
    fn should_inline_for_callee(&self, def_id: DefId) -> bool;

    fn check_codegen_attributes_extra(
        &self,
        callee_attrs: &CodegenFnAttrs,
    ) -> Result<(), &'static str>;

    fn check_caller_mir_body(&self, body: &Body<'tcx>) -> bool;

    /// Returns inlining decision that is based on the examination of callee MIR body.
    /// Assumes that codegen attributes have been checked for compatibility already.
    fn check_callee_mir_body(
        &self,
        callsite: &CallSite<'tcx>,
        callee_body: &Body<'tcx>,
        callee_attrs: &CodegenFnAttrs,
    ) -> Result<(), &'static str>;

    /// Called when inlining succeeds.
    fn on_inline_success(
        &mut self,
        callsite: &CallSite<'tcx>,
        caller_body: &mut Body<'tcx>,
        new_blocks: std::ops::Range<BasicBlock>,
    );

    /// Called when inlining failed or was not performed.
    fn on_inline_failure(&self, callsite: &CallSite<'tcx>, reason: &'static str);
}

struct ForceInliner<'tcx> {
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    /// `DefId` of caller.
    def_id: DefId,
    /// Stack of inlined instances.
    /// We only check the `DefId` and not the args because we want to
    /// avoid inlining cases of polymorphic recursion.
    /// The number of `DefId`s is finite, so checking history is enough
    /// to ensure that we do not loop endlessly while inlining.
    history: Vec<DefId>,
    /// Indicates that the caller body has been modified.
    changed: bool,
}

impl<'tcx> Inliner<'tcx> for ForceInliner<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, def_id: DefId, body: &Body<'tcx>) -> Self {
        Self { tcx, typing_env: body.typing_env(tcx), def_id, history: Vec::new(), changed: false }
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        self.typing_env
    }

    fn history(&self) -> &[DefId] {
        &self.history
    }

    fn caller_def_id(&self) -> DefId {
        self.def_id
    }

    fn changed(self) -> bool {
        self.changed
    }

    fn should_inline_for_callee(&self, def_id: DefId) -> bool {
        ForceInline::should_run_pass_for_callee(self.tcx(), def_id)
    }

    fn check_codegen_attributes_extra(
        &self,
        callee_attrs: &CodegenFnAttrs,
    ) -> Result<(), &'static str> {
        debug_assert_matches!(callee_attrs.inline, InlineAttr::Force { .. });
        Ok(())
    }

    fn check_caller_mir_body(&self, _: &Body<'tcx>) -> bool {
        true
    }

    #[instrument(level = "debug", skip(self, callee_body))]
    fn check_callee_mir_body(
        &self,
        _: &CallSite<'tcx>,
        callee_body: &Body<'tcx>,
        callee_attrs: &CodegenFnAttrs,
    ) -> Result<(), &'static str> {
        if callee_body.tainted_by_errors.is_some() {
            return Err("body has errors");
        }

        let caller_attrs = self.tcx().codegen_fn_attrs(self.caller_def_id());
        if callee_attrs.instruction_set != caller_attrs.instruction_set
            && callee_body
                .basic_blocks
                .iter()
                .any(|bb| matches!(bb.terminator().kind, TerminatorKind::InlineAsm { .. }))
        {
            // During the attribute checking stage we allow a callee with no
            // instruction_set assigned to count as compatible with a function that does
            // assign one. However, during this stage we require an exact match when any
            // inline-asm is detected. LLVM will still possibly do an inline later on
            // if the no-attribute function ends up with the same instruction set anyway.
            Err("cannot move inline-asm across instruction sets")
        } else {
            Ok(())
        }
    }

    fn on_inline_success(
        &mut self,
        callsite: &CallSite<'tcx>,
        caller_body: &mut Body<'tcx>,
        new_blocks: std::ops::Range<BasicBlock>,
    ) {
        self.changed = true;

        self.history.push(callsite.callee.def_id());
        process_blocks(self, caller_body, new_blocks);
        self.history.pop();
    }

    fn on_inline_failure(&self, callsite: &CallSite<'tcx>, reason: &'static str) {
        let tcx = self.tcx();
        let InlineAttr::Force { attr_span, reason: justification } =
            tcx.codegen_fn_attrs(callsite.callee.def_id()).inline
        else {
            bug!("called on item without required inlining");
        };

        let call_span = callsite.source_info.span;
        tcx.dcx().emit_err(crate::errors::ForceInlineFailure {
            call_span,
            attr_span,
            caller_span: tcx.def_span(self.def_id),
            caller: tcx.def_path_str(self.def_id),
            callee_span: tcx.def_span(callsite.callee.def_id()),
            callee: tcx.def_path_str(callsite.callee.def_id()),
            reason,
            justification: justification.map(|sym| crate::errors::ForceInlineJustification { sym }),
        });
    }
}

struct NormalInliner<'tcx> {
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    /// `DefId` of caller.
    def_id: DefId,
    /// Stack of inlined instances.
    /// We only check the `DefId` and not the args because we want to
    /// avoid inlining cases of polymorphic recursion.
    /// The number of `DefId`s is finite, so checking history is enough
    /// to ensure that we do not loop endlessly while inlining.
    history: Vec<DefId>,
    /// How many (multi-call) callsites have we inlined for the top-level call?
    ///
    /// We need to limit this in order to prevent super-linear growth in MIR size.
    top_down_counter: usize,
    /// Indicates that the caller body has been modified.
    changed: bool,
    /// Indicates that the caller is #[inline] and just calls another function,
    /// and thus we can inline less into it as it'll be inlined itself.
    caller_is_inline_forwarder: bool,
}

impl<'tcx> NormalInliner<'tcx> {
    fn past_depth_limit(&self) -> bool {
        self.history.len() > HISTORY_DEPTH_LIMIT || self.top_down_counter > TOP_DOWN_DEPTH_LIMIT
    }
}

impl<'tcx> Inliner<'tcx> for NormalInliner<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, def_id: DefId, body: &Body<'tcx>) -> Self {
        let typing_env = body.typing_env(tcx);
        let codegen_fn_attrs = tcx.codegen_fn_attrs(def_id);

        Self {
            tcx,
            typing_env,
            def_id,
            history: Vec::new(),
            top_down_counter: 0,
            changed: false,
            caller_is_inline_forwarder: matches!(
                codegen_fn_attrs.inline,
                InlineAttr::Hint | InlineAttr::Always | InlineAttr::Force { .. }
            ) && body_is_forwarder(body),
        }
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn caller_def_id(&self) -> DefId {
        self.def_id
    }

    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        self.typing_env
    }

    fn history(&self) -> &[DefId] {
        &self.history
    }

    fn changed(self) -> bool {
        self.changed
    }

    fn should_inline_for_callee(&self, _: DefId) -> bool {
        true
    }

    fn check_codegen_attributes_extra(
        &self,
        callee_attrs: &CodegenFnAttrs,
    ) -> Result<(), &'static str> {
        if self.past_depth_limit() && matches!(callee_attrs.inline, InlineAttr::None) {
            Err("Past depth limit so not inspecting unmarked callee")
        } else {
            Ok(())
        }
    }

    fn check_caller_mir_body(&self, body: &Body<'tcx>) -> bool {
        // Avoid inlining into coroutines, since their `optimized_mir` is used for layout computation,
        // which can create a cycle, even when no attempt is made to inline the function in the other
        // direction.
        if body.coroutine.is_some() {
            return false;
        }

        true
    }

    #[instrument(level = "debug", skip(self, callee_body))]
    fn check_callee_mir_body(
        &self,
        callsite: &CallSite<'tcx>,
        callee_body: &Body<'tcx>,
        callee_attrs: &CodegenFnAttrs,
    ) -> Result<(), &'static str> {
        let tcx = self.tcx();

        if let Some(_) = callee_body.tainted_by_errors {
            return Err("body has errors");
        }

        if self.past_depth_limit() && callee_body.basic_blocks.len() > 1 {
            return Err("Not inlining multi-block body as we're past a depth limit");
        }

        let mut threshold = if self.caller_is_inline_forwarder || self.past_depth_limit() {
            tcx.sess.opts.unstable_opts.inline_mir_forwarder_threshold.unwrap_or(30)
        } else if tcx.cross_crate_inlinable(callsite.callee.def_id()) {
            tcx.sess.opts.unstable_opts.inline_mir_hint_threshold.unwrap_or(100)
        } else {
            tcx.sess.opts.unstable_opts.inline_mir_threshold.unwrap_or(50)
        };

        // Give a bonus functions with a small number of blocks,
        // We normally have two or three blocks for even
        // very small functions.
        if callee_body.basic_blocks.len() <= 3 {
            threshold += threshold / 4;
        }
        debug!("    final inline threshold = {}", threshold);

        // FIXME: Give a bonus to functions with only a single caller

        let mut checker =
            CostChecker::new(tcx, self.typing_env(), Some(callsite.callee), callee_body);

        checker.add_function_level_costs();

        // Traverse the MIR manually so we can account for the effects of inlining on the CFG.
        let mut work_list = vec![START_BLOCK];
        let mut visited = DenseBitSet::new_empty(callee_body.basic_blocks.len());
        while let Some(bb) = work_list.pop() {
            if !visited.insert(bb.index()) {
                continue;
            }

            let blk = &callee_body.basic_blocks[bb];
            checker.visit_basic_block_data(bb, blk);

            let term = blk.terminator();
            let caller_attrs = tcx.codegen_fn_attrs(self.caller_def_id());
            if let TerminatorKind::Drop {
                ref place,
                target,
                unwind,
                replace: _,
                drop: _,
                async_fut: _,
            } = term.kind
            {
                work_list.push(target);

                // If the place doesn't actually need dropping, treat it like a regular goto.
                let ty = callsite
                    .callee
                    .instantiate_mir(tcx, ty::EarlyBinder::bind(&place.ty(callee_body, tcx).ty));
                if ty.needs_drop(tcx, self.typing_env())
                    && let UnwindAction::Cleanup(unwind) = unwind
                {
                    work_list.push(unwind);
                }
            } else if callee_attrs.instruction_set != caller_attrs.instruction_set
                && matches!(term.kind, TerminatorKind::InlineAsm { .. })
            {
                // During the attribute checking stage we allow a callee with no
                // instruction_set assigned to count as compatible with a function that does
                // assign one. However, during this stage we require an exact match when any
                // inline-asm is detected. LLVM will still possibly do an inline later on
                // if the no-attribute function ends up with the same instruction set anyway.
                return Err("cannot move inline-asm across instruction sets");
            } else if let TerminatorKind::TailCall { .. } = term.kind {
                // FIXME(explicit_tail_calls): figure out how exactly functions containing tail
                // calls can be inlined (and if they even should)
                return Err("can't inline functions with tail calls");
            } else {
                work_list.extend(term.successors())
            }
        }

        // N.B. We still apply our cost threshold to #[inline(always)] functions.
        // That attribute is often applied to very large functions that exceed LLVM's (very
        // generous) inlining threshold. Such functions are very poor MIR inlining candidates.
        // Always inlining #[inline(always)] functions in MIR, on net, slows down the compiler.
        let cost = checker.cost();
        if cost <= threshold {
            debug!("INLINING {:?} [cost={} <= threshold={}]", callsite, cost, threshold);
            Ok(())
        } else {
            debug!("NOT inlining {:?} [cost={} > threshold={}]", callsite, cost, threshold);
            Err("cost above threshold")
        }
    }

    fn on_inline_success(
        &mut self,
        callsite: &CallSite<'tcx>,
        caller_body: &mut Body<'tcx>,
        new_blocks: std::ops::Range<BasicBlock>,
    ) {
        self.changed = true;

        let new_calls_count = new_blocks
            .clone()
            .filter(|&bb| is_call_like(caller_body.basic_blocks[bb].terminator()))
            .count();
        if new_calls_count > 1 {
            self.top_down_counter += 1;
        }

        self.history.push(callsite.callee.def_id());
        process_blocks(self, caller_body, new_blocks);
        self.history.pop();

        if self.history.is_empty() {
            self.top_down_counter = 0;
        }
    }

    fn on_inline_failure(&self, _: &CallSite<'tcx>, _: &'static str) {}
}

fn inline<'tcx, T: Inliner<'tcx>>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) -> bool {
    let def_id = body.source.def_id();

    // Only do inlining into fn bodies.
    if !tcx.hir_body_owner_kind(def_id).is_fn_or_closure() {
        return false;
    }

    let mut inliner = T::new(tcx, def_id, body);
    if !inliner.check_caller_mir_body(body) {
        return false;
    }

    let blocks = START_BLOCK..body.basic_blocks.next_index();
    process_blocks(&mut inliner, body, blocks);
    inliner.changed()
}

fn process_blocks<'tcx, I: Inliner<'tcx>>(
    inliner: &mut I,
    caller_body: &mut Body<'tcx>,
    blocks: Range<BasicBlock>,
) {
    for bb in blocks {
        let bb_data = &caller_body[bb];
        if bb_data.is_cleanup {
            continue;
        }

        let Some(callsite) = resolve_callsite(inliner, caller_body, bb, bb_data) else {
            continue;
        };

        let span = trace_span!("process_blocks", %callsite.callee, ?bb);
        let _guard = span.enter();

        match try_inlining(inliner, caller_body, &callsite) {
            Err(reason) => {
                debug!("not-inlined {} [{}]", callsite.callee, reason);
                inliner.on_inline_failure(&callsite, reason);
            }
            Ok(new_blocks) => {
                debug!("inlined {}", callsite.callee);
                inliner.on_inline_success(&callsite, caller_body, new_blocks);
            }
        }
    }
}

fn resolve_callsite<'tcx, I: Inliner<'tcx>>(
    inliner: &I,
    caller_body: &Body<'tcx>,
    bb: BasicBlock,
    bb_data: &BasicBlockData<'tcx>,
) -> Option<CallSite<'tcx>> {
    let tcx = inliner.tcx();
    // Only consider direct calls to functions
    let terminator = bb_data.terminator();

    // FIXME(explicit_tail_calls): figure out if we can inline tail calls
    if let TerminatorKind::Call { ref func, fn_span, .. } = terminator.kind {
        let func_ty = func.ty(caller_body, tcx);
        if let ty::FnDef(def_id, args) = *func_ty.kind() {
            if !inliner.should_inline_for_callee(def_id) {
                debug!("not enabled");
                return None;
            }

            // To resolve an instance its args have to be fully normalized.
            let args = tcx.try_normalize_erasing_regions(inliner.typing_env(), args).ok()?;
            let callee =
                Instance::try_resolve(tcx, inliner.typing_env(), def_id, args).ok().flatten()?;

            if let InstanceKind::Virtual(..) | InstanceKind::Intrinsic(_) = callee.def {
                return None;
            }

            if inliner.history().contains(&callee.def_id()) {
                return None;
            }

            let fn_sig = tcx.fn_sig(def_id).instantiate(tcx, args);

            // Additionally, check that the body that we're inlining actually agrees
            // with the ABI of the trait that the item comes from.
            if let InstanceKind::Item(instance_def_id) = callee.def
                && tcx.def_kind(instance_def_id) == DefKind::AssocFn
                && let instance_fn_sig = tcx.fn_sig(instance_def_id).skip_binder()
                && instance_fn_sig.abi() != fn_sig.abi()
            {
                return None;
            }

            let source_info = SourceInfo { span: fn_span, ..terminator.source_info };

            return Some(CallSite { callee, fn_sig, block: bb, source_info });
        }
    }

    None
}

/// Attempts to inline a callsite into the caller body. When successful returns basic blocks
/// containing the inlined body. Otherwise returns an error describing why inlining didn't take
/// place.
fn try_inlining<'tcx, I: Inliner<'tcx>>(
    inliner: &I,
    caller_body: &mut Body<'tcx>,
    callsite: &CallSite<'tcx>,
) -> Result<std::ops::Range<BasicBlock>, &'static str> {
    let tcx = inliner.tcx();
    check_mir_is_available(inliner, caller_body, callsite.callee)?;

    let callee_attrs = tcx.codegen_fn_attrs(callsite.callee.def_id());
    check_inline::is_inline_valid_on_fn(tcx, callsite.callee.def_id())?;
    check_codegen_attributes(inliner, callsite, callee_attrs)?;
    inliner.check_codegen_attributes_extra(callee_attrs)?;

    let terminator = caller_body[callsite.block].terminator.as_ref().unwrap();
    let TerminatorKind::Call { args, destination, .. } = &terminator.kind else { bug!() };
    let destination_ty = destination.ty(&caller_body.local_decls, tcx).ty;
    for arg in args {
        if !arg.node.ty(&caller_body.local_decls, tcx).is_sized(tcx, inliner.typing_env()) {
            // We do not allow inlining functions with unsized params. Inlining these functions
            // could create unsized locals, which are unsound and being phased out.
            return Err("call has unsized argument");
        }
    }

    let callee_body = try_instance_mir(tcx, callsite.callee.def)?;
    check_inline::is_inline_valid_on_body(tcx, callee_body)?;
    inliner.check_callee_mir_body(callsite, callee_body, callee_attrs)?;

    let Ok(callee_body) = callsite.callee.try_instantiate_mir_and_normalize_erasing_regions(
        tcx,
        inliner.typing_env(),
        ty::EarlyBinder::bind(callee_body.clone()),
    ) else {
        debug!("failed to normalize callee body");
        return Err("implementation limitation -- could not normalize callee body");
    };

    // Normally, this shouldn't be required, but trait normalization failure can create a
    // validation ICE.
    if !validate_types(tcx, inliner.typing_env(), &callee_body, &caller_body).is_empty() {
        debug!("failed to validate callee body");
        return Err("implementation limitation -- callee body failed validation");
    }

    // Check call signature compatibility.
    // Normally, this shouldn't be required, but trait normalization failure can create a
    // validation ICE.
    let output_type = callee_body.return_ty();
    if !util::sub_types(tcx, inliner.typing_env(), output_type, destination_ty) {
        trace!(?output_type, ?destination_ty);
        return Err("implementation limitation -- return type mismatch");
    }
    if callsite.fn_sig.abi() == ExternAbi::RustCall {
        let (self_arg, arg_tuple) = match &args[..] {
            [arg_tuple] => (None, arg_tuple),
            [self_arg, arg_tuple] => (Some(self_arg), arg_tuple),
            _ => bug!("Expected `rust-call` to have 1 or 2 args"),
        };

        let self_arg_ty = self_arg.map(|self_arg| self_arg.node.ty(&caller_body.local_decls, tcx));

        let arg_tuple_ty = arg_tuple.node.ty(&caller_body.local_decls, tcx);
        let arg_tys = if callee_body.spread_arg.is_some() {
            std::slice::from_ref(&arg_tuple_ty)
        } else {
            let ty::Tuple(arg_tuple_tys) = *arg_tuple_ty.kind() else {
                bug!("Closure arguments are not passed as a tuple");
            };
            arg_tuple_tys.as_slice()
        };

        for (arg_ty, input) in
            self_arg_ty.into_iter().chain(arg_tys.iter().copied()).zip(callee_body.args_iter())
        {
            let input_type = callee_body.local_decls[input].ty;
            if !util::sub_types(tcx, inliner.typing_env(), input_type, arg_ty) {
                trace!(?arg_ty, ?input_type);
                debug!("failed to normalize tuple argument type");
                return Err("implementation limitation");
            }
        }
    } else {
        for (arg, input) in args.iter().zip(callee_body.args_iter()) {
            let input_type = callee_body.local_decls[input].ty;
            let arg_ty = arg.node.ty(&caller_body.local_decls, tcx);
            if !util::sub_types(tcx, inliner.typing_env(), input_type, arg_ty) {
                trace!(?arg_ty, ?input_type);
                debug!("failed to normalize argument type");
                return Err("implementation limitation -- arg mismatch");
            }
        }
    }

    let old_blocks = caller_body.basic_blocks.next_index();
    inline_call(inliner, caller_body, callsite, callee_body);
    let new_blocks = old_blocks..caller_body.basic_blocks.next_index();

    Ok(new_blocks)
}

fn check_mir_is_available<'tcx, I: Inliner<'tcx>>(
    inliner: &I,
    caller_body: &Body<'tcx>,
    callee: Instance<'tcx>,
) -> Result<(), &'static str> {
    let caller_def_id = caller_body.source.def_id();
    let callee_def_id = callee.def_id();
    if callee_def_id == caller_def_id {
        return Err("self-recursion");
    }

    match callee.def {
        InstanceKind::Item(_) => {
            // If there is no MIR available (either because it was not in metadata or
            // because it has no MIR because it's an extern function), then the inliner
            // won't cause cycles on this.
            if !inliner.tcx().is_mir_available(callee_def_id) {
                debug!("item MIR unavailable");
                return Err("implementation limitation -- MIR unavailable");
            }
        }
        // These have no own callable MIR.
        InstanceKind::Intrinsic(_) | InstanceKind::Virtual(..) => {
            debug!("instance without MIR (intrinsic / virtual)");
            return Err("implementation limitation -- cannot inline intrinsic");
        }

        // FIXME(#127030): `ConstParamHasTy` has bad interactions with
        // the drop shim builder, which does not evaluate predicates in
        // the correct param-env for types being dropped. Stall resolving
        // the MIR for this instance until all of its const params are
        // substituted.
        InstanceKind::DropGlue(_, Some(ty)) if ty.has_type_flags(TypeFlags::HAS_CT_PARAM) => {
            debug!("still needs substitution");
            return Err("implementation limitation -- HACK for dropping polymorphic type");
        }
        InstanceKind::AsyncDropGlue(_, ty) | InstanceKind::AsyncDropGlueCtorShim(_, ty) => {
            return if ty.still_further_specializable() {
                Err("still needs substitution")
            } else {
                Ok(())
            };
        }
        InstanceKind::FutureDropPollShim(_, ty, ty2) => {
            return if ty.still_further_specializable() || ty2.still_further_specializable() {
                Err("still needs substitution")
            } else {
                Ok(())
            };
        }

        // This cannot result in an immediate cycle since the callee MIR is a shim, which does
        // not get any optimizations run on it. Any subsequent inlining may cause cycles, but we
        // do not need to catch this here, we can wait until the inliner decides to continue
        // inlining a second time.
        InstanceKind::VTableShim(_)
        | InstanceKind::ReifyShim(..)
        | InstanceKind::FnPtrShim(..)
        | InstanceKind::ClosureOnceShim { .. }
        | InstanceKind::ConstructCoroutineInClosureShim { .. }
        | InstanceKind::DropGlue(..)
        | InstanceKind::CloneShim(..)
        | InstanceKind::ThreadLocalShim(..)
        | InstanceKind::FnPtrAddrShim(..) => return Ok(()),
    }

    if inliner.tcx().is_constructor(callee_def_id) {
        trace!("constructors always have MIR");
        // Constructor functions cannot cause a query cycle.
        return Ok(());
    }

    if let Some(callee_def_id) = callee_def_id.as_local()
        && !inliner
            .tcx()
            .is_lang_item(inliner.tcx().parent(caller_def_id), rustc_hir::LangItem::FnOnce)
    {
        // If we know for sure that the function we're calling will itself try to
        // call us, then we avoid inlining that function.
        if inliner.tcx().mir_callgraph_cyclic(caller_def_id.expect_local()).contains(&callee_def_id)
        {
            debug!("query cycle avoidance");
            return Err("caller might be reachable from callee");
        }

        Ok(())
    } else {
        // This cannot result in an immediate cycle since the callee MIR is from another crate
        // and is already optimized. Any subsequent inlining may cause cycles, but we do
        // not need to catch this here, we can wait until the inliner decides to continue
        // inlining a second time.
        trace!("functions from other crates always have MIR");
        Ok(())
    }
}

/// Returns an error if inlining is not possible based on codegen attributes alone. A success
/// indicates that inlining decision should be based on other criteria.
fn check_codegen_attributes<'tcx, I: Inliner<'tcx>>(
    inliner: &I,
    callsite: &CallSite<'tcx>,
    callee_attrs: &CodegenFnAttrs,
) -> Result<(), &'static str> {
    let tcx = inliner.tcx();
    if let InlineAttr::Never = callee_attrs.inline {
        return Err("never inline attribute");
    }

    if let OptimizeAttr::DoNotOptimize = callee_attrs.optimize {
        return Err("has DoNotOptimize attribute");
    }

    inliner.check_codegen_attributes_extra(callee_attrs)?;

    // Reachability pass defines which functions are eligible for inlining. Generally inlining
    // other functions is incorrect because they could reference symbols that aren't exported.
    let is_generic = callsite.callee.args.non_erasable_generics().next().is_some();
    if !is_generic && !tcx.cross_crate_inlinable(callsite.callee.def_id()) {
        return Err("not exported");
    }

    let codegen_fn_attrs = tcx.codegen_fn_attrs(inliner.caller_def_id());
    if callee_attrs.no_sanitize != codegen_fn_attrs.no_sanitize {
        return Err("incompatible sanitizer set");
    }

    // Two functions are compatible if the callee has no attribute (meaning
    // that it's codegen agnostic), or sets an attribute that is identical
    // to this function's attribute.
    if callee_attrs.instruction_set.is_some()
        && callee_attrs.instruction_set != codegen_fn_attrs.instruction_set
    {
        return Err("incompatible instruction set");
    }

    let callee_feature_names = callee_attrs.target_features.iter().map(|f| f.name);
    let this_feature_names = codegen_fn_attrs.target_features.iter().map(|f| f.name);
    if callee_feature_names.ne(this_feature_names) {
        // In general it is not correct to inline a callee with target features that are a
        // subset of the caller. This is because the callee might contain calls, and the ABI of
        // those calls depends on the target features of the surrounding function. By moving a
        // `Call` terminator from one MIR body to another with more target features, we might
        // change the ABI of that call!
        return Err("incompatible target features");
    }

    Ok(())
}

fn inline_call<'tcx, I: Inliner<'tcx>>(
    inliner: &I,
    caller_body: &mut Body<'tcx>,
    callsite: &CallSite<'tcx>,
    mut callee_body: Body<'tcx>,
) {
    let tcx = inliner.tcx();
    let terminator = caller_body[callsite.block].terminator.take().unwrap();
    let TerminatorKind::Call { func, args, destination, unwind, target, .. } = terminator.kind
    else {
        bug!("unexpected terminator kind {:?}", terminator.kind);
    };

    let return_block = if let Some(block) = target {
        // Prepare a new block for code that should execute when call returns. We don't use
        // target block directly since it might have other predecessors.
        let data = BasicBlockData::new(
            Some(Terminator {
                source_info: terminator.source_info,
                kind: TerminatorKind::Goto { target: block },
            }),
            caller_body[block].is_cleanup,
        );
        Some(caller_body.basic_blocks_mut().push(data))
    } else {
        None
    };

    // If the call is something like `a[*i] = f(i)`, where
    // `i : &mut usize`, then just duplicating the `a[*i]`
    // Place could result in two different locations if `f`
    // writes to `i`. To prevent this we need to create a temporary
    // borrow of the place and pass the destination as `*temp` instead.
    fn dest_needs_borrow(place: Place<'_>) -> bool {
        for elem in place.projection.iter() {
            match elem {
                ProjectionElem::Deref | ProjectionElem::Index(_) => return true,
                _ => {}
            }
        }

        false
    }

    let dest = if dest_needs_borrow(destination) {
        trace!("creating temp for return destination");
        let dest = Rvalue::Ref(
            tcx.lifetimes.re_erased,
            BorrowKind::Mut { kind: MutBorrowKind::Default },
            destination,
        );
        let dest_ty = dest.ty(caller_body, tcx);
        let temp = Place::from(new_call_temp(caller_body, callsite, dest_ty, return_block));
        caller_body[callsite.block].statements.push(Statement::new(
            callsite.source_info,
            StatementKind::Assign(Box::new((temp, dest))),
        ));
        tcx.mk_place_deref(temp)
    } else {
        destination
    };

    // Always create a local to hold the destination, as `RETURN_PLACE` may appear
    // where a full `Place` is not allowed.
    let (remap_destination, destination_local) = if let Some(d) = dest.as_local() {
        (false, d)
    } else {
        (
            true,
            new_call_temp(caller_body, callsite, destination.ty(caller_body, tcx).ty, return_block),
        )
    };

    // Copy the arguments if needed.
    let args = make_call_args(inliner, args, callsite, caller_body, &callee_body, return_block);

    let mut integrator = Integrator {
        args: &args,
        new_locals: caller_body.local_decls.next_index()..,
        new_scopes: caller_body.source_scopes.next_index()..,
        new_blocks: caller_body.basic_blocks.next_index()..,
        destination: destination_local,
        callsite_scope: caller_body.source_scopes[callsite.source_info.scope].clone(),
        callsite,
        cleanup_block: unwind,
        in_cleanup_block: false,
        return_block,
        tcx,
        always_live_locals: UsedInStmtLocals::new(&callee_body).locals,
    };

    // Map all `Local`s, `SourceScope`s and `BasicBlock`s to new ones
    // (or existing ones, in a few special cases) in the caller.
    integrator.visit_body(&mut callee_body);

    // If there are any locals without storage markers, give them storage only for the
    // duration of the call.
    for local in callee_body.vars_and_temps_iter() {
        if integrator.always_live_locals.contains(local) {
            let new_local = integrator.map_local(local);
            caller_body[callsite.block]
                .statements
                .push(Statement::new(callsite.source_info, StatementKind::StorageLive(new_local)));
        }
    }
    if let Some(block) = return_block {
        // To avoid repeated O(n) insert, push any new statements to the end and rotate
        // the slice once.
        let mut n = 0;
        if remap_destination {
            caller_body[block].statements.push(Statement::new(
                callsite.source_info,
                StatementKind::Assign(Box::new((
                    dest,
                    Rvalue::Use(Operand::Move(destination_local.into())),
                ))),
            ));
            n += 1;
        }
        for local in callee_body.vars_and_temps_iter().rev() {
            if integrator.always_live_locals.contains(local) {
                let new_local = integrator.map_local(local);
                caller_body[block].statements.push(Statement::new(
                    callsite.source_info,
                    StatementKind::StorageDead(new_local),
                ));
                n += 1;
            }
        }
        caller_body[block].statements.rotate_right(n);
    }

    // Insert all of the (mapped) parts of the callee body into the caller.
    caller_body.local_decls.extend(callee_body.drain_vars_and_temps());
    caller_body.source_scopes.append(&mut callee_body.source_scopes);

    // only "full" debug promises any variable-level information
    if tcx
        .sess
        .opts
        .unstable_opts
        .inline_mir_preserve_debug
        .unwrap_or(tcx.sess.opts.debuginfo == DebugInfo::Full)
    {
        // -Zinline-mir-preserve-debug is enabled when building the standard library, so that
        // people working on rust can build with or without debuginfo while
        // still getting consistent results from the mir-opt tests.
        caller_body.var_debug_info.append(&mut callee_body.var_debug_info);
    } else {
        for bb in callee_body.basic_blocks_mut() {
            bb.drop_debuginfo();
        }
    }
    caller_body.basic_blocks_mut().append(callee_body.basic_blocks_mut());

    caller_body[callsite.block].terminator = Some(Terminator {
        source_info: callsite.source_info,
        kind: TerminatorKind::Goto { target: integrator.map_block(START_BLOCK) },
    });

    // Copy required constants from the callee_body into the caller_body. Although we are only
    // pushing unevaluated consts to `required_consts`, here they may have been evaluated
    // because we are calling `instantiate_and_normalize_erasing_regions` -- so we filter again.
    caller_body.required_consts.as_mut().unwrap().extend(
        callee_body.required_consts().into_iter().filter(|ct| ct.const_.is_required_const()),
    );
    // Now that we incorporated the callee's `required_consts`, we can remove the callee from
    // `mentioned_items` -- but we have to take their `mentioned_items` in return. This does
    // some extra work here to save the monomorphization collector work later. It helps a lot,
    // since monomorphization can avoid a lot of work when the "mentioned items" are similar to
    // the actually used items. By doing this we can entirely avoid visiting the callee!
    // We need to reconstruct the `required_item` for the callee so that we can find and
    // remove it.
    let callee_item = MentionedItem::Fn(func.ty(caller_body, tcx));
    let caller_mentioned_items = caller_body.mentioned_items.as_mut().unwrap();
    if let Some(idx) = caller_mentioned_items.iter().position(|item| item.node == callee_item) {
        // We found the callee, so remove it and add its items instead.
        caller_mentioned_items.remove(idx);
        caller_mentioned_items.extend(callee_body.mentioned_items());
    } else {
        // If we can't find the callee, there's no point in adding its items. Probably it
        // already got removed by being inlined elsewhere in the same function, so we already
        // took its items.
    }
}

fn make_call_args<'tcx, I: Inliner<'tcx>>(
    inliner: &I,
    args: Box<[Spanned<Operand<'tcx>>]>,
    callsite: &CallSite<'tcx>,
    caller_body: &mut Body<'tcx>,
    callee_body: &Body<'tcx>,
    return_block: Option<BasicBlock>,
) -> Box<[Local]> {
    let tcx = inliner.tcx();

    // There is a bit of a mismatch between the *caller* of a closure and the *callee*.
    // The caller provides the arguments wrapped up in a tuple:
    //
    //     tuple_tmp = (a, b, c)
    //     Fn::call(closure_ref, tuple_tmp)
    //
    // meanwhile the closure body expects the arguments (here, `a`, `b`, and `c`)
    // as distinct arguments. (This is the "rust-call" ABI hack.) Normally, codegen has
    // the job of unpacking this tuple. But here, we are codegen. =) So we want to create
    // a vector like
    //
    //     [closure_ref, tuple_tmp.0, tuple_tmp.1, tuple_tmp.2]
    //
    // Except for one tiny wrinkle: we don't actually want `tuple_tmp.0`. It's more convenient
    // if we "spill" that into *another* temporary, so that we can map the argument
    // variable in the callee MIR directly to an argument variable on our side.
    // So we introduce temporaries like:
    //
    //     tmp0 = tuple_tmp.0
    //     tmp1 = tuple_tmp.1
    //     tmp2 = tuple_tmp.2
    //
    // and the vector is `[closure_ref, tmp0, tmp1, tmp2]`.
    if callsite.fn_sig.abi() == ExternAbi::RustCall && callee_body.spread_arg.is_none() {
        // FIXME(edition_2024): switch back to a normal method call.
        let mut args = <_>::into_iter(args);
        let self_ = create_temp_if_necessary(
            inliner,
            args.next().unwrap().node,
            callsite,
            caller_body,
            return_block,
        );
        let tuple = create_temp_if_necessary(
            inliner,
            args.next().unwrap().node,
            callsite,
            caller_body,
            return_block,
        );
        assert!(args.next().is_none());

        let tuple = Place::from(tuple);
        let ty::Tuple(tuple_tys) = tuple.ty(caller_body, tcx).ty.kind() else {
            bug!("Closure arguments are not passed as a tuple");
        };

        // The `closure_ref` in our example above.
        let closure_ref_arg = iter::once(self_);

        // The `tmp0`, `tmp1`, and `tmp2` in our example above.
        let tuple_tmp_args = tuple_tys.iter().enumerate().map(|(i, ty)| {
            // This is e.g., `tuple_tmp.0` in our example above.
            let tuple_field = Operand::Move(tcx.mk_place_field(tuple, FieldIdx::new(i), ty));

            // Spill to a local to make e.g., `tmp0`.
            create_temp_if_necessary(inliner, tuple_field, callsite, caller_body, return_block)
        });

        closure_ref_arg.chain(tuple_tmp_args).collect()
    } else {
        args.into_iter()
            .map(|a| create_temp_if_necessary(inliner, a.node, callsite, caller_body, return_block))
            .collect()
    }
}

/// If `arg` is already a temporary, returns it. Otherwise, introduces a fresh temporary `T` and an
/// instruction `T = arg`, and returns `T`.
fn create_temp_if_necessary<'tcx, I: Inliner<'tcx>>(
    inliner: &I,
    arg: Operand<'tcx>,
    callsite: &CallSite<'tcx>,
    caller_body: &mut Body<'tcx>,
    return_block: Option<BasicBlock>,
) -> Local {
    // Reuse the operand if it is a moved temporary.
    if let Operand::Move(place) = &arg
        && let Some(local) = place.as_local()
        && caller_body.local_kind(local) == LocalKind::Temp
    {
        return local;
    }

    // Otherwise, create a temporary for the argument.
    trace!("creating temp for argument {:?}", arg);
    let arg_ty = arg.ty(caller_body, inliner.tcx());
    let local = new_call_temp(caller_body, callsite, arg_ty, return_block);
    caller_body[callsite.block].statements.push(Statement::new(
        callsite.source_info,
        StatementKind::Assign(Box::new((Place::from(local), Rvalue::Use(arg)))),
    ));
    local
}

/// Introduces a new temporary into the caller body that is live for the duration of the call.
fn new_call_temp<'tcx>(
    caller_body: &mut Body<'tcx>,
    callsite: &CallSite<'tcx>,
    ty: Ty<'tcx>,
    return_block: Option<BasicBlock>,
) -> Local {
    let local = caller_body.local_decls.push(LocalDecl::new(ty, callsite.source_info.span));

    caller_body[callsite.block]
        .statements
        .push(Statement::new(callsite.source_info, StatementKind::StorageLive(local)));

    if let Some(block) = return_block {
        caller_body[block]
            .statements
            .insert(0, Statement::new(callsite.source_info, StatementKind::StorageDead(local)));
    }

    local
}

/**
 * Integrator.
 *
 * Integrates blocks from the callee function into the calling function.
 * Updates block indices, references to locals and other control flow
 * stuff.
*/
struct Integrator<'a, 'tcx> {
    args: &'a [Local],
    new_locals: RangeFrom<Local>,
    new_scopes: RangeFrom<SourceScope>,
    new_blocks: RangeFrom<BasicBlock>,
    destination: Local,
    callsite_scope: SourceScopeData<'tcx>,
    callsite: &'a CallSite<'tcx>,
    cleanup_block: UnwindAction,
    in_cleanup_block: bool,
    return_block: Option<BasicBlock>,
    tcx: TyCtxt<'tcx>,
    always_live_locals: DenseBitSet<Local>,
}

impl Integrator<'_, '_> {
    fn map_local(&self, local: Local) -> Local {
        let new = if local == RETURN_PLACE {
            self.destination
        } else {
            let idx = local.index() - 1;
            if idx < self.args.len() {
                self.args[idx]
            } else {
                self.new_locals.start + (idx - self.args.len())
            }
        };
        trace!("mapping local `{:?}` to `{:?}`", local, new);
        new
    }

    fn map_scope(&self, scope: SourceScope) -> SourceScope {
        let new = self.new_scopes.start + scope.index();
        trace!("mapping scope `{:?}` to `{:?}`", scope, new);
        new
    }

    fn map_block(&self, block: BasicBlock) -> BasicBlock {
        let new = self.new_blocks.start + block.index();
        trace!("mapping block `{:?}` to `{:?}`", block, new);
        new
    }

    fn map_unwind(&self, unwind: UnwindAction) -> UnwindAction {
        if self.in_cleanup_block {
            match unwind {
                UnwindAction::Cleanup(_) | UnwindAction::Continue => {
                    bug!("cleanup on cleanup block");
                }
                UnwindAction::Unreachable | UnwindAction::Terminate(_) => return unwind,
            }
        }

        match unwind {
            UnwindAction::Unreachable | UnwindAction::Terminate(_) => unwind,
            UnwindAction::Cleanup(target) => UnwindAction::Cleanup(self.map_block(target)),
            // Add an unwind edge to the original call's cleanup block
            UnwindAction::Continue => self.cleanup_block,
        }
    }
}

impl<'tcx> MutVisitor<'tcx> for Integrator<'_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_local(&mut self, local: &mut Local, _ctxt: PlaceContext, _location: Location) {
        *local = self.map_local(*local);
    }

    fn visit_source_scope_data(&mut self, scope_data: &mut SourceScopeData<'tcx>) {
        self.super_source_scope_data(scope_data);
        if scope_data.parent_scope.is_none() {
            // Attach the outermost callee scope as a child of the callsite
            // scope, via the `parent_scope` and `inlined_parent_scope` chains.
            scope_data.parent_scope = Some(self.callsite.source_info.scope);
            assert_eq!(scope_data.inlined_parent_scope, None);
            scope_data.inlined_parent_scope = if self.callsite_scope.inlined.is_some() {
                Some(self.callsite.source_info.scope)
            } else {
                self.callsite_scope.inlined_parent_scope
            };

            // Mark the outermost callee scope as an inlined one.
            assert_eq!(scope_data.inlined, None);
            scope_data.inlined = Some((self.callsite.callee, self.callsite.source_info.span));
        } else if scope_data.inlined_parent_scope.is_none() {
            // Make it easy to find the scope with `inlined` set above.
            scope_data.inlined_parent_scope = Some(self.map_scope(OUTERMOST_SOURCE_SCOPE));
        }
    }

    fn visit_source_scope(&mut self, scope: &mut SourceScope) {
        *scope = self.map_scope(*scope);
    }

    fn visit_basic_block_data(&mut self, block: BasicBlock, data: &mut BasicBlockData<'tcx>) {
        self.in_cleanup_block = data.is_cleanup;
        self.super_basic_block_data(block, data);
        self.in_cleanup_block = false;
    }

    fn visit_retag(&mut self, kind: &mut RetagKind, place: &mut Place<'tcx>, loc: Location) {
        self.super_retag(kind, place, loc);

        // We have to patch all inlined retags to be aware that they are no longer
        // happening on function entry.
        if *kind == RetagKind::FnEntry {
            *kind = RetagKind::Default;
        }
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        if let StatementKind::StorageLive(local) | StatementKind::StorageDead(local) =
            statement.kind
        {
            self.always_live_locals.remove(local);
        }
        self.super_statement(statement, location);
    }

    fn visit_terminator(&mut self, terminator: &mut Terminator<'tcx>, loc: Location) {
        // Don't try to modify the implicit `_0` access on return (`return` terminators are
        // replaced down below anyways).
        if !matches!(terminator.kind, TerminatorKind::Return) {
            self.super_terminator(terminator, loc);
        } else {
            self.visit_source_info(&mut terminator.source_info);
        }

        match terminator.kind {
            TerminatorKind::CoroutineDrop | TerminatorKind::Yield { .. } => bug!(),
            TerminatorKind::Goto { ref mut target } => {
                *target = self.map_block(*target);
            }
            TerminatorKind::SwitchInt { ref mut targets, .. } => {
                for tgt in targets.all_targets_mut() {
                    *tgt = self.map_block(*tgt);
                }
            }
            TerminatorKind::Drop { ref mut target, ref mut unwind, .. } => {
                *target = self.map_block(*target);
                *unwind = self.map_unwind(*unwind);
            }
            TerminatorKind::TailCall { .. } => {
                // check_mir_body forbids tail calls
                unreachable!()
            }
            TerminatorKind::Call { ref mut target, ref mut unwind, .. } => {
                if let Some(ref mut tgt) = *target {
                    *tgt = self.map_block(*tgt);
                }
                *unwind = self.map_unwind(*unwind);
            }
            TerminatorKind::Assert { ref mut target, ref mut unwind, .. } => {
                *target = self.map_block(*target);
                *unwind = self.map_unwind(*unwind);
            }
            TerminatorKind::Return => {
                terminator.kind = if let Some(tgt) = self.return_block {
                    TerminatorKind::Goto { target: tgt }
                } else {
                    TerminatorKind::Unreachable
                }
            }
            TerminatorKind::UnwindResume => {
                terminator.kind = match self.cleanup_block {
                    UnwindAction::Cleanup(tgt) => TerminatorKind::Goto { target: tgt },
                    UnwindAction::Continue => TerminatorKind::UnwindResume,
                    UnwindAction::Unreachable => TerminatorKind::Unreachable,
                    UnwindAction::Terminate(reason) => TerminatorKind::UnwindTerminate(reason),
                };
            }
            TerminatorKind::UnwindTerminate(_) => {}
            TerminatorKind::Unreachable => {}
            TerminatorKind::FalseEdge { ref mut real_target, ref mut imaginary_target } => {
                *real_target = self.map_block(*real_target);
                *imaginary_target = self.map_block(*imaginary_target);
            }
            TerminatorKind::FalseUnwind { real_target: _, unwind: _ } =>
            // see the ordering of passes in the optimized_mir query.
            {
                bug!("False unwinds should have been removed before inlining")
            }
            TerminatorKind::InlineAsm { ref mut targets, ref mut unwind, .. } => {
                for tgt in targets.iter_mut() {
                    *tgt = self.map_block(*tgt);
                }
                *unwind = self.map_unwind(*unwind);
            }
        }
    }
}

#[instrument(skip(tcx), level = "debug")]
fn try_instance_mir<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: InstanceKind<'tcx>,
) -> Result<&'tcx Body<'tcx>, &'static str> {
    if let ty::InstanceKind::DropGlue(_, Some(ty)) | ty::InstanceKind::AsyncDropGlueCtorShim(_, ty) =
        instance
        && let ty::Adt(def, args) = ty.kind()
    {
        let fields = def.all_fields();
        for field in fields {
            let field_ty = field.ty(tcx, args);
            if field_ty.has_param() && field_ty.has_aliases() {
                return Err("cannot build drop shim for polymorphic type");
            }
        }
    }
    Ok(tcx.instance_mir(instance))
}

fn body_is_forwarder(body: &Body<'_>) -> bool {
    let TerminatorKind::Call { target, .. } = body.basic_blocks[START_BLOCK].terminator().kind
    else {
        return false;
    };
    if let Some(target) = target {
        let TerminatorKind::Return = body.basic_blocks[target].terminator().kind else {
            return false;
        };
    }

    let max_blocks = if !body.is_polymorphic {
        2
    } else if target.is_none() {
        3
    } else {
        4
    };
    if body.basic_blocks.len() > max_blocks {
        return false;
    }

    body.basic_blocks.iter_enumerated().all(|(bb, bb_data)| {
        bb == START_BLOCK
            || matches!(
                bb_data.terminator().kind,
                TerminatorKind::Return
                    | TerminatorKind::Drop { .. }
                    | TerminatorKind::UnwindResume
                    | TerminatorKind::UnwindTerminate(_)
            )
    })
}
