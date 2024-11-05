//! Inlining pass for MIR functions.

use std::iter;
use std::ops::{Range, RangeFrom};

use rustc_abi::{ExternAbi, FieldIdx};
use rustc_hir::InlineAttr;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_index::Idx;
use rustc_index::bit_set::BitSet;
use rustc_middle::bug;
use rustc_middle::middle::codegen_fn_attrs::{CodegenFnAttrFlags, CodegenFnAttrs};
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Instance, InstanceKind, Ty, TyCtxt, TypeFlags, TypeVisitableExt};
use rustc_session::config::{DebugInfo, OptLevel};
use rustc_span::source_map::Spanned;
use rustc_span::sym;
use tracing::{debug, instrument, trace, trace_span};

use crate::cost_checker::CostChecker;
use crate::deref_separator::deref_finder;
use crate::simplify::simplify_cfg;
use crate::util;
use crate::validate::validate_types;

pub(crate) mod cycle;

const TOP_DOWN_DEPTH_LIMIT: usize = 5;

// Made public so that `mir_drops_elaborated_and_const_checked` can be overridden
// by custom rustc drivers, running all the steps by themselves. See #114628.
pub struct Inline;

#[derive(Clone, Debug)]
struct CallSite<'tcx> {
    callee: Instance<'tcx>,
    fn_sig: ty::PolyFnSig<'tcx>,
    block: BasicBlock,
    source_info: SourceInfo,
}

impl<'tcx> crate::MirPass<'tcx> for Inline {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        // FIXME(#127234): Coverage instrumentation currently doesn't handle inlined
        // MIR correctly when Modified Condition/Decision Coverage is enabled.
        if sess.instrument_coverage_mcdc() {
            return false;
        }

        if let Some(enabled) = sess.opts.unstable_opts.inline_mir {
            return enabled;
        }

        match sess.mir_opt_level() {
            0 | 1 => false,
            2 => {
                (sess.opts.optimize == OptLevel::Default
                    || sess.opts.optimize == OptLevel::Aggressive)
                    && sess.opts.incremental == None
            }
            _ => true,
        }
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let span = trace_span!("inline", body = %tcx.def_path_str(body.source.def_id()));
        let _guard = span.enter();
        if inline(tcx, body) {
            debug!("running simplify cfg on {:?}", body.source);
            simplify_cfg(body);
            deref_finder(tcx, body);
        }
    }
}

fn inline<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) -> bool {
    let def_id = body.source.def_id().expect_local();

    // Only do inlining into fn bodies.
    if !tcx.hir().body_owner_kind(def_id).is_fn_or_closure() {
        return false;
    }
    if body.source.promoted.is_some() {
        return false;
    }
    // Avoid inlining into coroutines, since their `optimized_mir` is used for layout computation,
    // which can create a cycle, even when no attempt is made to inline the function in the other
    // direction.
    if body.coroutine.is_some() {
        return false;
    }

    let typing_env = body.typing_env(tcx);
    let codegen_fn_attrs = tcx.codegen_fn_attrs(def_id);

    let mut this = Inliner {
        tcx,
        typing_env,
        codegen_fn_attrs,
        history: Vec::new(),
        changed: false,
        caller_is_inline_forwarder: matches!(
            codegen_fn_attrs.inline,
            InlineAttr::Hint | InlineAttr::Always
        ) && body_is_forwarder(body),
    };
    let blocks = START_BLOCK..body.basic_blocks.next_index();
    this.process_blocks(body, blocks);
    this.changed
}

struct Inliner<'tcx> {
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    /// Caller codegen attributes.
    codegen_fn_attrs: &'tcx CodegenFnAttrs,
    /// Stack of inlined instances.
    /// We only check the `DefId` and not the args because we want to
    /// avoid inlining cases of polymorphic recursion.
    /// The number of `DefId`s is finite, so checking history is enough
    /// to ensure that we do not loop endlessly while inlining.
    history: Vec<DefId>,
    /// Indicates that the caller body has been modified.
    changed: bool,
    /// Indicates that the caller is #[inline] and just calls another function,
    /// and thus we can inline less into it as it'll be inlined itself.
    caller_is_inline_forwarder: bool,
}

impl<'tcx> Inliner<'tcx> {
    fn process_blocks(&mut self, caller_body: &mut Body<'tcx>, blocks: Range<BasicBlock>) {
        // How many callsites in this body are we allowed to inline? We need to limit this in order
        // to prevent super-linear growth in MIR size
        let inline_limit = match self.history.len() {
            0 => usize::MAX,
            1..=TOP_DOWN_DEPTH_LIMIT => 1,
            _ => return,
        };
        let mut inlined_count = 0;
        for bb in blocks {
            let bb_data = &caller_body[bb];
            if bb_data.is_cleanup {
                continue;
            }

            let Some(callsite) = self.resolve_callsite(caller_body, bb, bb_data) else {
                continue;
            };

            let span = trace_span!("process_blocks", %callsite.callee, ?bb);
            let _guard = span.enter();

            match self.try_inlining(caller_body, &callsite) {
                Err(reason) => {
                    debug!("not-inlined {} [{}]", callsite.callee, reason);
                }
                Ok(new_blocks) => {
                    debug!("inlined {}", callsite.callee);
                    self.changed = true;

                    self.history.push(callsite.callee.def_id());
                    self.process_blocks(caller_body, new_blocks);
                    self.history.pop();

                    inlined_count += 1;
                    if inlined_count == inline_limit {
                        debug!("inline count reached");
                        return;
                    }
                }
            }
        }
    }

    /// Attempts to inline a callsite into the caller body. When successful returns basic blocks
    /// containing the inlined body. Otherwise returns an error describing why inlining didn't take
    /// place.
    fn try_inlining(
        &self,
        caller_body: &mut Body<'tcx>,
        callsite: &CallSite<'tcx>,
    ) -> Result<std::ops::Range<BasicBlock>, &'static str> {
        self.check_mir_is_available(caller_body, callsite.callee)?;

        let callee_attrs = self.tcx.codegen_fn_attrs(callsite.callee.def_id());
        let cross_crate_inlinable = self.tcx.cross_crate_inlinable(callsite.callee.def_id());
        self.check_codegen_attributes(callsite, callee_attrs, cross_crate_inlinable)?;

        // Intrinsic fallback bodies are automatically made cross-crate inlineable,
        // but at this stage we don't know whether codegen knows the intrinsic,
        // so just conservatively don't inline it.
        if self.tcx.has_attr(callsite.callee.def_id(), sym::rustc_intrinsic) {
            return Err("Callee is an intrinsic, do not inline fallback bodies");
        }

        let terminator = caller_body[callsite.block].terminator.as_ref().unwrap();
        let TerminatorKind::Call { args, destination, .. } = &terminator.kind else { bug!() };
        let destination_ty = destination.ty(&caller_body.local_decls, self.tcx).ty;
        for arg in args {
            if !arg.node.ty(&caller_body.local_decls, self.tcx).is_sized(self.tcx, self.typing_env)
            {
                // We do not allow inlining functions with unsized params. Inlining these functions
                // could create unsized locals, which are unsound and being phased out.
                return Err("Call has unsized argument");
            }
        }

        let callee_body = try_instance_mir(self.tcx, callsite.callee.def)?;
        self.check_mir_body(callsite, callee_body, callee_attrs, cross_crate_inlinable)?;

        let Ok(callee_body) = callsite.callee.try_instantiate_mir_and_normalize_erasing_regions(
            self.tcx,
            self.typing_env,
            ty::EarlyBinder::bind(callee_body.clone()),
        ) else {
            return Err("failed to normalize callee body");
        };

        // Normally, this shouldn't be required, but trait normalization failure can create a
        // validation ICE.
        if !validate_types(self.tcx, self.typing_env, &callee_body, &caller_body).is_empty() {
            return Err("failed to validate callee body");
        }

        // Check call signature compatibility.
        // Normally, this shouldn't be required, but trait normalization failure can create a
        // validation ICE.
        let output_type = callee_body.return_ty();
        if !util::sub_types(self.tcx, self.typing_env, output_type, destination_ty) {
            trace!(?output_type, ?destination_ty);
            return Err("failed to normalize return type");
        }
        if callsite.fn_sig.abi() == ExternAbi::RustCall {
            // FIXME: Don't inline user-written `extern "rust-call"` functions,
            // since this is generally perf-negative on rustc, and we hope that
            // LLVM will inline these functions instead.
            if callee_body.spread_arg.is_some() {
                return Err("do not inline user-written rust-call functions");
            }

            let (self_arg, arg_tuple) = match &args[..] {
                [arg_tuple] => (None, arg_tuple),
                [self_arg, arg_tuple] => (Some(self_arg), arg_tuple),
                _ => bug!("Expected `rust-call` to have 1 or 2 args"),
            };

            let self_arg_ty =
                self_arg.map(|self_arg| self_arg.node.ty(&caller_body.local_decls, self.tcx));

            let arg_tuple_ty = arg_tuple.node.ty(&caller_body.local_decls, self.tcx);
            let ty::Tuple(arg_tuple_tys) = *arg_tuple_ty.kind() else {
                bug!("Closure arguments are not passed as a tuple");
            };

            for (arg_ty, input) in
                self_arg_ty.into_iter().chain(arg_tuple_tys).zip(callee_body.args_iter())
            {
                let input_type = callee_body.local_decls[input].ty;
                if !util::sub_types(self.tcx, self.typing_env, input_type, arg_ty) {
                    trace!(?arg_ty, ?input_type);
                    return Err("failed to normalize tuple argument type");
                }
            }
        } else {
            for (arg, input) in args.iter().zip(callee_body.args_iter()) {
                let input_type = callee_body.local_decls[input].ty;
                let arg_ty = arg.node.ty(&caller_body.local_decls, self.tcx);
                if !util::sub_types(self.tcx, self.typing_env, input_type, arg_ty) {
                    trace!(?arg_ty, ?input_type);
                    return Err("failed to normalize argument type");
                }
            }
        }

        let old_blocks = caller_body.basic_blocks.next_index();
        self.inline_call(caller_body, callsite, callee_body);
        let new_blocks = old_blocks..caller_body.basic_blocks.next_index();

        Ok(new_blocks)
    }

    fn check_mir_is_available(
        &self,
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
                if !self.tcx.is_mir_available(callee_def_id) {
                    return Err("item MIR unavailable");
                }
            }
            // These have no own callable MIR.
            InstanceKind::Intrinsic(_) | InstanceKind::Virtual(..) => {
                return Err("instance without MIR (intrinsic / virtual)");
            }

            // FIXME(#127030): `ConstParamHasTy` has bad interactions with
            // the drop shim builder, which does not evaluate predicates in
            // the correct param-env for types being dropped. Stall resolving
            // the MIR for this instance until all of its const params are
            // substituted.
            InstanceKind::DropGlue(_, Some(ty)) if ty.has_type_flags(TypeFlags::HAS_CT_PARAM) => {
                return Err("still needs substitution");
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
            | InstanceKind::FnPtrAddrShim(..)
            | InstanceKind::AsyncDropGlueCtorShim(..) => return Ok(()),
        }

        if self.tcx.is_constructor(callee_def_id) {
            trace!("constructors always have MIR");
            // Constructor functions cannot cause a query cycle.
            return Ok(());
        }

        if callee_def_id.is_local() {
            // If we know for sure that the function we're calling will itself try to
            // call us, then we avoid inlining that function.
            if self.tcx.mir_callgraph_reachable((callee, caller_def_id.expect_local())) {
                return Err("caller might be reachable from callee (query cycle avoidance)");
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

    fn resolve_callsite(
        &self,
        caller_body: &Body<'tcx>,
        bb: BasicBlock,
        bb_data: &BasicBlockData<'tcx>,
    ) -> Option<CallSite<'tcx>> {
        // Only consider direct calls to functions
        let terminator = bb_data.terminator();

        // FIXME(explicit_tail_calls): figure out if we can inline tail calls
        if let TerminatorKind::Call { ref func, fn_span, .. } = terminator.kind {
            let func_ty = func.ty(caller_body, self.tcx);
            if let ty::FnDef(def_id, args) = *func_ty.kind() {
                // To resolve an instance its args have to be fully normalized.
                let args = self.tcx.try_normalize_erasing_regions(self.typing_env, args).ok()?;
                let callee = Instance::try_resolve(self.tcx, self.typing_env, def_id, args)
                    .ok()
                    .flatten()?;

                if let InstanceKind::Virtual(..) | InstanceKind::Intrinsic(_) = callee.def {
                    return None;
                }

                if self.history.contains(&callee.def_id()) {
                    return None;
                }

                let fn_sig = self.tcx.fn_sig(def_id).instantiate(self.tcx, args);

                // Additionally, check that the body that we're inlining actually agrees
                // with the ABI of the trait that the item comes from.
                if let InstanceKind::Item(instance_def_id) = callee.def
                    && self.tcx.def_kind(instance_def_id) == DefKind::AssocFn
                    && let instance_fn_sig = self.tcx.fn_sig(instance_def_id).skip_binder()
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

    /// Returns an error if inlining is not possible based on codegen attributes alone. A success
    /// indicates that inlining decision should be based on other criteria.
    fn check_codegen_attributes(
        &self,
        callsite: &CallSite<'tcx>,
        callee_attrs: &CodegenFnAttrs,
        cross_crate_inlinable: bool,
    ) -> Result<(), &'static str> {
        if self.tcx.has_attr(callsite.callee.def_id(), sym::rustc_no_mir_inline) {
            return Err("#[rustc_no_mir_inline]");
        }

        if let InlineAttr::Never = callee_attrs.inline {
            return Err("never inline hint");
        }

        // Reachability pass defines which functions are eligible for inlining. Generally inlining
        // other functions is incorrect because they could reference symbols that aren't exported.
        let is_generic = callsite.callee.args.non_erasable_generics().next().is_some();
        if !is_generic && !cross_crate_inlinable {
            return Err("not exported");
        }

        if callsite.fn_sig.c_variadic() {
            return Err("C variadic");
        }

        if callee_attrs.flags.contains(CodegenFnAttrFlags::COLD) {
            return Err("cold");
        }

        if callee_attrs.no_sanitize != self.codegen_fn_attrs.no_sanitize {
            return Err("incompatible sanitizer set");
        }

        // Two functions are compatible if the callee has no attribute (meaning
        // that it's codegen agnostic), or sets an attribute that is identical
        // to this function's attribute.
        if callee_attrs.instruction_set.is_some()
            && callee_attrs.instruction_set != self.codegen_fn_attrs.instruction_set
        {
            return Err("incompatible instruction set");
        }

        let callee_feature_names = callee_attrs.target_features.iter().map(|f| f.name);
        let this_feature_names = self.codegen_fn_attrs.target_features.iter().map(|f| f.name);
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

    /// Returns inlining decision that is based on the examination of callee MIR body.
    /// Assumes that codegen attributes have been checked for compatibility already.
    #[instrument(level = "debug", skip(self, callee_body))]
    fn check_mir_body(
        &self,
        callsite: &CallSite<'tcx>,
        callee_body: &Body<'tcx>,
        callee_attrs: &CodegenFnAttrs,
        cross_crate_inlinable: bool,
    ) -> Result<(), &'static str> {
        let tcx = self.tcx;

        if let Some(_) = callee_body.tainted_by_errors {
            return Err("Body is tainted");
        }

        let mut threshold = if self.caller_is_inline_forwarder {
            self.tcx.sess.opts.unstable_opts.inline_mir_forwarder_threshold.unwrap_or(30)
        } else if cross_crate_inlinable {
            self.tcx.sess.opts.unstable_opts.inline_mir_hint_threshold.unwrap_or(100)
        } else {
            self.tcx.sess.opts.unstable_opts.inline_mir_threshold.unwrap_or(50)
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
            CostChecker::new(self.tcx, self.typing_env, Some(callsite.callee), callee_body);

        checker.add_function_level_costs();

        // Traverse the MIR manually so we can account for the effects of inlining on the CFG.
        let mut work_list = vec![START_BLOCK];
        let mut visited = BitSet::new_empty(callee_body.basic_blocks.len());
        while let Some(bb) = work_list.pop() {
            if !visited.insert(bb.index()) {
                continue;
            }

            let blk = &callee_body.basic_blocks[bb];
            checker.visit_basic_block_data(bb, blk);

            let term = blk.terminator();
            if let TerminatorKind::Drop { ref place, target, unwind, replace: _ } = term.kind {
                work_list.push(target);

                // If the place doesn't actually need dropping, treat it like a regular goto.
                let ty = callsite.callee.instantiate_mir(
                    self.tcx,
                    ty::EarlyBinder::bind(&place.ty(callee_body, tcx).ty),
                );
                if ty.needs_drop(tcx, self.typing_env)
                    && let UnwindAction::Cleanup(unwind) = unwind
                {
                    work_list.push(unwind);
                }
            } else if callee_attrs.instruction_set != self.codegen_fn_attrs.instruction_set
                && matches!(term.kind, TerminatorKind::InlineAsm { .. })
            {
                // During the attribute checking stage we allow a callee with no
                // instruction_set assigned to count as compatible with a function that does
                // assign one. However, during this stage we require an exact match when any
                // inline-asm is detected. LLVM will still possibly do an inline later on
                // if the no-attribute function ends up with the same instruction set anyway.
                return Err("Cannot move inline-asm across instruction sets");
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

    fn inline_call(
        &self,
        caller_body: &mut Body<'tcx>,
        callsite: &CallSite<'tcx>,
        mut callee_body: Body<'tcx>,
    ) {
        let terminator = caller_body[callsite.block].terminator.take().unwrap();
        let TerminatorKind::Call { func, args, destination, unwind, target, .. } = terminator.kind
        else {
            bug!("unexpected terminator kind {:?}", terminator.kind);
        };

        let return_block = if let Some(block) = target {
            // Prepare a new block for code that should execute when call returns. We don't use
            // target block directly since it might have other predecessors.
            let mut data = BasicBlockData::new(Some(Terminator {
                source_info: terminator.source_info,
                kind: TerminatorKind::Goto { target: block },
            }));
            data.is_cleanup = caller_body[block].is_cleanup;
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
                self.tcx.lifetimes.re_erased,
                BorrowKind::Mut { kind: MutBorrowKind::Default },
                destination,
            );
            let dest_ty = dest.ty(caller_body, self.tcx);
            let temp =
                Place::from(self.new_call_temp(caller_body, callsite, dest_ty, return_block));
            caller_body[callsite.block].statements.push(Statement {
                source_info: callsite.source_info,
                kind: StatementKind::Assign(Box::new((temp, dest))),
            });
            self.tcx.mk_place_deref(temp)
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
                self.new_call_temp(
                    caller_body,
                    callsite,
                    destination.ty(caller_body, self.tcx).ty,
                    return_block,
                ),
            )
        };

        // Copy the arguments if needed.
        let args = self.make_call_args(args, callsite, caller_body, &callee_body, return_block);

        let mut integrator = Integrator {
            args: &args,
            new_locals: Local::new(caller_body.local_decls.len())..,
            new_scopes: SourceScope::new(caller_body.source_scopes.len())..,
            new_blocks: BasicBlock::new(caller_body.basic_blocks.len())..,
            destination: destination_local,
            callsite_scope: caller_body.source_scopes[callsite.source_info.scope].clone(),
            callsite,
            cleanup_block: unwind,
            in_cleanup_block: false,
            return_block,
            tcx: self.tcx,
            always_live_locals: BitSet::new_filled(callee_body.local_decls.len()),
        };

        // Map all `Local`s, `SourceScope`s and `BasicBlock`s to new ones
        // (or existing ones, in a few special cases) in the caller.
        integrator.visit_body(&mut callee_body);

        // If there are any locals without storage markers, give them storage only for the
        // duration of the call.
        for local in callee_body.vars_and_temps_iter() {
            if integrator.always_live_locals.contains(local) {
                let new_local = integrator.map_local(local);
                caller_body[callsite.block].statements.push(Statement {
                    source_info: callsite.source_info,
                    kind: StatementKind::StorageLive(new_local),
                });
            }
        }
        if let Some(block) = return_block {
            // To avoid repeated O(n) insert, push any new statements to the end and rotate
            // the slice once.
            let mut n = 0;
            if remap_destination {
                caller_body[block].statements.push(Statement {
                    source_info: callsite.source_info,
                    kind: StatementKind::Assign(Box::new((
                        dest,
                        Rvalue::Use(Operand::Move(destination_local.into())),
                    ))),
                });
                n += 1;
            }
            for local in callee_body.vars_and_temps_iter().rev() {
                if integrator.always_live_locals.contains(local) {
                    let new_local = integrator.map_local(local);
                    caller_body[block].statements.push(Statement {
                        source_info: callsite.source_info,
                        kind: StatementKind::StorageDead(new_local),
                    });
                    n += 1;
                }
            }
            caller_body[block].statements.rotate_right(n);
        }

        // Insert all of the (mapped) parts of the callee body into the caller.
        caller_body.local_decls.extend(callee_body.drain_vars_and_temps());
        caller_body.source_scopes.append(&mut callee_body.source_scopes);
        if self
            .tcx
            .sess
            .opts
            .unstable_opts
            .inline_mir_preserve_debug
            .unwrap_or(self.tcx.sess.opts.debuginfo != DebugInfo::None)
        {
            // Note that we need to preserve these in the standard library so that
            // people working on rust can build with or without debuginfo while
            // still getting consistent results from the mir-opt tests.
            caller_body.var_debug_info.append(&mut callee_body.var_debug_info);
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
        let callee_item = MentionedItem::Fn(func.ty(caller_body, self.tcx));
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

    fn make_call_args(
        &self,
        args: Box<[Spanned<Operand<'tcx>>]>,
        callsite: &CallSite<'tcx>,
        caller_body: &mut Body<'tcx>,
        callee_body: &Body<'tcx>,
        return_block: Option<BasicBlock>,
    ) -> Box<[Local]> {
        let tcx = self.tcx;

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
            let self_ = self.create_temp_if_necessary(
                args.next().unwrap().node,
                callsite,
                caller_body,
                return_block,
            );
            let tuple = self.create_temp_if_necessary(
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
                self.create_temp_if_necessary(tuple_field, callsite, caller_body, return_block)
            });

            closure_ref_arg.chain(tuple_tmp_args).collect()
        } else {
            // FIXME(edition_2024): switch back to a normal method call.
            <_>::into_iter(args)
                .map(|a| self.create_temp_if_necessary(a.node, callsite, caller_body, return_block))
                .collect()
        }
    }

    /// If `arg` is already a temporary, returns it. Otherwise, introduces a fresh
    /// temporary `T` and an instruction `T = arg`, and returns `T`.
    fn create_temp_if_necessary(
        &self,
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
        let arg_ty = arg.ty(caller_body, self.tcx);
        let local = self.new_call_temp(caller_body, callsite, arg_ty, return_block);
        caller_body[callsite.block].statements.push(Statement {
            source_info: callsite.source_info,
            kind: StatementKind::Assign(Box::new((Place::from(local), Rvalue::Use(arg)))),
        });
        local
    }

    /// Introduces a new temporary into the caller body that is live for the duration of the call.
    fn new_call_temp(
        &self,
        caller_body: &mut Body<'tcx>,
        callsite: &CallSite<'tcx>,
        ty: Ty<'tcx>,
        return_block: Option<BasicBlock>,
    ) -> Local {
        let local = caller_body.local_decls.push(LocalDecl::new(ty, callsite.source_info.span));

        caller_body[callsite.block].statements.push(Statement {
            source_info: callsite.source_info,
            kind: StatementKind::StorageLive(local),
        });

        if let Some(block) = return_block {
            caller_body[block].statements.insert(0, Statement {
                source_info: callsite.source_info,
                kind: StatementKind::StorageDead(local),
            });
        }

        local
    }
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
    always_live_locals: BitSet<Local>,
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
                Local::new(self.new_locals.start.index() + (idx - self.args.len()))
            }
        };
        trace!("mapping local `{:?}` to `{:?}`", local, new);
        new
    }

    fn map_scope(&self, scope: SourceScope) -> SourceScope {
        let new = SourceScope::new(self.new_scopes.start.index() + scope.index());
        trace!("mapping scope `{:?}` to `{:?}`", scope, new);
        new
    }

    fn map_block(&self, block: BasicBlock) -> BasicBlock {
        let new = BasicBlock::new(self.new_blocks.start.index() + block.index());
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
    if let ty::InstanceKind::DropGlue(_, Some(ty))
    | ty::InstanceKind::AsyncDropGlueCtorShim(_, Some(ty)) = instance
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
