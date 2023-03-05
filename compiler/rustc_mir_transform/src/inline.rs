//! Inlining pass for MIR functions
use crate::deref_separator::deref_finder;
use rustc_attr::InlineAttr;
use rustc_hir::def_id::DefId;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::Idx;
use rustc_middle::middle::codegen_fn_attrs::{CodegenFnAttrFlags, CodegenFnAttrs};
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Instance, InstanceDef, ParamEnv, Ty, TyCtxt};
use rustc_session::config::OptLevel;
use rustc_span::{hygiene::ExpnKind, ExpnData, LocalExpnId, Span};
use rustc_target::abi::VariantIdx;
use rustc_target::spec::abi::Abi;

use crate::simplify::{remove_dead_blocks, CfgSimplifier};
use crate::util;
use crate::MirPass;
use std::iter;
use std::ops::{Range, RangeFrom};

pub(crate) mod cycle;

const INSTR_COST: usize = 5;
const CALL_PENALTY: usize = 25;
const LANDINGPAD_PENALTY: usize = 50;
const RESUME_PENALTY: usize = 45;

const UNKNOWN_SIZE_COST: usize = 10;

const TOP_DOWN_DEPTH_LIMIT: usize = 5;

pub struct Inline;

#[derive(Copy, Clone, Debug)]
struct CallSite<'tcx> {
    callee: Instance<'tcx>,
    fn_sig: ty::PolyFnSig<'tcx>,
    block: BasicBlock,
    target: Option<BasicBlock>,
    source_info: SourceInfo,
}

impl<'tcx> MirPass<'tcx> for Inline {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
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
            CfgSimplifier::new(body).simplify();
            remove_dead_blocks(tcx, body);
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
    // Avoid inlining into generators, since their `optimized_mir` is used for layout computation,
    // which can create a cycle, even when no attempt is made to inline the function in the other
    // direction.
    if body.generator.is_some() {
        return false;
    }

    let param_env = tcx.param_env_reveal_all_normalized(def_id);

    let mut this = Inliner {
        tcx,
        param_env,
        codegen_fn_attrs: tcx.codegen_fn_attrs(def_id),
        history: Vec::new(),
        changed: false,
    };
    let blocks = START_BLOCK..body.basic_blocks.next_index();
    this.process_blocks(body, blocks);
    this.changed
}

struct Inliner<'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    /// Caller codegen attributes.
    codegen_fn_attrs: &'tcx CodegenFnAttrs,
    /// Stack of inlined instances.
    /// We only check the `DefId` and not the substs because we want to
    /// avoid inlining cases of polymorphic recursion.
    /// The number of `DefId`s is finite, so checking history is enough
    /// to ensure that we do not loop endlessly while inlining.
    history: Vec<DefId>,
    /// Indicates that the caller body has been modified.
    changed: bool,
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
                    continue;
                }
                Ok(new_blocks) => {
                    debug!("inlined {}", callsite.callee);
                    self.changed = true;
                    inlined_count += 1;
                    if inlined_count == inline_limit {
                        return;
                    }
                    self.history.push(callsite.callee.def_id());
                    self.process_blocks(caller_body, new_blocks);
                    self.history.pop();
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
        let callee_attrs = self.tcx.codegen_fn_attrs(callsite.callee.def_id());
        self.check_codegen_attributes(callsite, callee_attrs)?;
        self.check_mir_is_available(caller_body, &callsite.callee)?;
        let callee_body = self.tcx.instance_mir(callsite.callee.def);
        self.check_mir_body(callsite, callee_body, callee_attrs)?;

        if !self.tcx.consider_optimizing(|| {
            format!("Inline {:?} into {:?}", callsite.callee, caller_body.source)
        }) {
            return Err("optimization fuel exhausted");
        }

        let Ok(callee_body) = callsite.callee.try_subst_mir_and_normalize_erasing_regions(
            self.tcx,
            self.param_env,
            callee_body.clone(),
        ) else {
            return Err("failed to normalize callee body");
        };

        // Check call signature compatibility.
        // Normally, this shouldn't be required, but trait normalization failure can create a
        // validation ICE.
        let terminator = caller_body[callsite.block].terminator.as_ref().unwrap();
        let TerminatorKind::Call { args, destination, .. } = &terminator.kind else { bug!() };
        let destination_ty = destination.ty(&caller_body.local_decls, self.tcx).ty;
        let output_type = callee_body.return_ty();
        if !util::is_subtype(self.tcx, self.param_env, output_type, destination_ty) {
            trace!(?output_type, ?destination_ty);
            return Err("failed to normalize return type");
        }
        if callsite.fn_sig.abi() == Abi::RustCall {
            let (arg_tuple, skipped_args) = match &args[..] {
                [arg_tuple] => (arg_tuple, 0),
                [_, arg_tuple] => (arg_tuple, 1),
                _ => bug!("Expected `rust-call` to have 1 or 2 args"),
            };

            let arg_tuple_ty = arg_tuple.ty(&caller_body.local_decls, self.tcx);
            let ty::Tuple(arg_tuple_tys) = arg_tuple_ty.kind() else {
                bug!("Closure arguments are not passed as a tuple");
            };

            for (arg_ty, input) in
                arg_tuple_tys.iter().zip(callee_body.args_iter().skip(skipped_args))
            {
                let input_type = callee_body.local_decls[input].ty;
                if !util::is_subtype(self.tcx, self.param_env, input_type, arg_ty) {
                    trace!(?arg_ty, ?input_type);
                    return Err("failed to normalize tuple argument type");
                }
            }
        } else {
            for (arg, input) in args.iter().zip(callee_body.args_iter()) {
                let input_type = callee_body.local_decls[input].ty;
                let arg_ty = arg.ty(&caller_body.local_decls, self.tcx);
                if !util::is_subtype(self.tcx, self.param_env, input_type, arg_ty) {
                    trace!(?arg_ty, ?input_type);
                    return Err("failed to normalize argument type");
                }
            }
        }

        let old_blocks = caller_body.basic_blocks.next_index();
        self.inline_call(caller_body, &callsite, callee_body);
        let new_blocks = old_blocks..caller_body.basic_blocks.next_index();

        Ok(new_blocks)
    }

    fn check_mir_is_available(
        &self,
        caller_body: &Body<'tcx>,
        callee: &Instance<'tcx>,
    ) -> Result<(), &'static str> {
        let caller_def_id = caller_body.source.def_id();
        let callee_def_id = callee.def_id();
        if callee_def_id == caller_def_id {
            return Err("self-recursion");
        }

        match callee.def {
            InstanceDef::Item(_) => {
                // If there is no MIR available (either because it was not in metadata or
                // because it has no MIR because it's an extern function), then the inliner
                // won't cause cycles on this.
                if !self.tcx.is_mir_available(callee_def_id) {
                    return Err("item MIR unavailable");
                }
            }
            // These have no own callable MIR.
            InstanceDef::Intrinsic(_) | InstanceDef::Virtual(..) => {
                return Err("instance without MIR (intrinsic / virtual)");
            }
            // This cannot result in an immediate cycle since the callee MIR is a shim, which does
            // not get any optimizations run on it. Any subsequent inlining may cause cycles, but we
            // do not need to catch this here, we can wait until the inliner decides to continue
            // inlining a second time.
            InstanceDef::VTableShim(_)
            | InstanceDef::ReifyShim(_)
            | InstanceDef::FnPtrShim(..)
            | InstanceDef::ClosureOnceShim { .. }
            | InstanceDef::DropGlue(..)
            | InstanceDef::CloneShim(..) => return Ok(()),
        }

        if self.tcx.is_constructor(callee_def_id) {
            trace!("constructors always have MIR");
            // Constructor functions cannot cause a query cycle.
            return Ok(());
        }

        if callee_def_id.is_local() {
            // Avoid a cycle here by only using `instance_mir` only if we have
            // a lower `DefPathHash` than the callee. This ensures that the callee will
            // not inline us. This trick even works with incremental compilation,
            // since `DefPathHash` is stable.
            if self.tcx.def_path_hash(caller_def_id).local_hash()
                < self.tcx.def_path_hash(callee_def_id).local_hash()
            {
                return Ok(());
            }

            // If we know for sure that the function we're calling will itself try to
            // call us, then we avoid inlining that function.
            if self.tcx.mir_callgraph_reachable((*callee, caller_def_id.expect_local())) {
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
        if let TerminatorKind::Call { ref func, target, fn_span, .. } = terminator.kind {
            let func_ty = func.ty(caller_body, self.tcx);
            if let ty::FnDef(def_id, substs) = *func_ty.kind() {
                // To resolve an instance its substs have to be fully normalized.
                let substs = self.tcx.try_normalize_erasing_regions(self.param_env, substs).ok()?;
                let callee =
                    Instance::resolve(self.tcx, self.param_env, def_id, substs).ok().flatten()?;

                if let InstanceDef::Virtual(..) | InstanceDef::Intrinsic(_) = callee.def {
                    return None;
                }

                if self.history.contains(&callee.def_id()) {
                    return None;
                }

                let fn_sig = self.tcx.fn_sig(def_id).subst(self.tcx, substs);
                let source_info = SourceInfo { span: fn_span, ..terminator.source_info };

                return Some(CallSite { callee, fn_sig, block: bb, target, source_info });
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
    ) -> Result<(), &'static str> {
        match callee_attrs.inline {
            InlineAttr::Never => return Err("never inline hint"),
            InlineAttr::Always | InlineAttr::Hint => {}
            InlineAttr::None => {
                if self.tcx.sess.mir_opt_level() <= 2 {
                    return Err("at mir-opt-level=2, only #[inline] is inlined");
                }
            }
        }

        // Only inline local functions if they would be eligible for cross-crate
        // inlining. This is to ensure that the final crate doesn't have MIR that
        // reference unexported symbols
        if callsite.callee.def_id().is_local() {
            let is_generic = callsite.callee.substs.non_erasable_generics().next().is_some();
            if !is_generic && !callee_attrs.requests_inline() {
                return Err("not exported");
            }
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

        for feature in &callee_attrs.target_features {
            if !self.codegen_fn_attrs.target_features.contains(feature) {
                return Err("incompatible target feature");
            }
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
    ) -> Result<(), &'static str> {
        let tcx = self.tcx;

        let mut threshold = if callee_attrs.requests_inline() {
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
        let diverges = matches!(
            callee_body.basic_blocks[START_BLOCK].terminator().kind,
            TerminatorKind::Unreachable | TerminatorKind::Call { target: None, .. }
        );
        if diverges && !matches!(callee_attrs.inline, InlineAttr::Always) {
            return Err("callee diverges unconditionally");
        }

        let mut checker = CostChecker {
            tcx: self.tcx,
            param_env: self.param_env,
            instance: callsite.callee,
            callee_body,
            cost: 0,
            validation: Ok(()),
        };

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
            if let TerminatorKind::Drop { ref place, target, unwind } = term.kind {
                work_list.push(target);

                // If the place doesn't actually need dropping, treat it like a regular goto.
                let ty = callsite.callee.subst_mir(self.tcx, &place.ty(callee_body, tcx).ty);
                if ty.needs_drop(tcx, self.param_env) && let Some(unwind) = unwind {
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
            } else {
                work_list.extend(term.successors())
            }
        }

        // Count up the cost of local variables and temps, if we know the size
        // use that, otherwise we use a moderately-large dummy cost.
        for v in callee_body.vars_and_temps_iter() {
            checker.visit_local_decl(v, &callee_body.local_decls[v]);
        }

        // Abort if type validation found anything fishy.
        checker.validation?;

        let cost = checker.cost;
        if let InlineAttr::Always = callee_attrs.inline {
            debug!("INLINING {:?} because inline(always) [cost={}]", callsite, cost);
            Ok(())
        } else if cost <= threshold {
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
        match terminator.kind {
            TerminatorKind::Call { args, destination, cleanup, .. } => {
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
                        BorrowKind::Mut { allow_two_phase_borrow: false },
                        destination,
                    );
                    let dest_ty = dest.ty(caller_body, self.tcx);
                    let temp = Place::from(self.new_call_temp(caller_body, &callsite, dest_ty));
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
                            &callsite,
                            destination.ty(caller_body, self.tcx).ty,
                        ),
                    )
                };

                // Copy the arguments if needed.
                let args: Vec<_> = self.make_call_args(args, &callsite, caller_body, &callee_body);

                let mut expn_data = ExpnData::default(
                    ExpnKind::Inlined,
                    callsite.source_info.span,
                    self.tcx.sess.edition(),
                    None,
                    None,
                );
                expn_data.def_site = callee_body.span;
                let expn_data =
                    self.tcx.with_stable_hashing_context(|hcx| LocalExpnId::fresh(expn_data, hcx));
                let mut integrator = Integrator {
                    args: &args,
                    new_locals: Local::new(caller_body.local_decls.len())..,
                    new_scopes: SourceScope::new(caller_body.source_scopes.len())..,
                    new_blocks: BasicBlock::new(caller_body.basic_blocks.len())..,
                    destination: destination_local,
                    callsite_scope: caller_body.source_scopes[callsite.source_info.scope].clone(),
                    callsite,
                    cleanup_block: cleanup,
                    in_cleanup_block: false,
                    tcx: self.tcx,
                    expn_data,
                    always_live_locals: BitSet::new_filled(callee_body.local_decls.len()),
                };

                // Map all `Local`s, `SourceScope`s and `BasicBlock`s to new ones
                // (or existing ones, in a few special cases) in the caller.
                integrator.visit_body(&mut callee_body);

                // If there are any locals without storage markers, give them storage only for the
                // duration of the call.
                for local in callee_body.vars_and_temps_iter() {
                    if !callee_body.local_decls[local].internal
                        && integrator.always_live_locals.contains(local)
                    {
                        let new_local = integrator.map_local(local);
                        caller_body[callsite.block].statements.push(Statement {
                            source_info: callsite.source_info,
                            kind: StatementKind::StorageLive(new_local),
                        });
                    }
                }
                if let Some(block) = callsite.target {
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
                        if !callee_body.local_decls[local].internal
                            && integrator.always_live_locals.contains(local)
                        {
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
                caller_body.source_scopes.extend(&mut callee_body.source_scopes.drain(..));
                caller_body.var_debug_info.append(&mut callee_body.var_debug_info);
                caller_body.basic_blocks_mut().extend(callee_body.basic_blocks_mut().drain(..));

                caller_body[callsite.block].terminator = Some(Terminator {
                    source_info: callsite.source_info,
                    kind: TerminatorKind::Goto { target: integrator.map_block(START_BLOCK) },
                });

                // Copy only unevaluated constants from the callee_body into the caller_body.
                // Although we are only pushing `ConstKind::Unevaluated` consts to
                // `required_consts`, here we may not only have `ConstKind::Unevaluated`
                // because we are calling `subst_and_normalize_erasing_regions`.
                caller_body.required_consts.extend(
                    callee_body.required_consts.iter().copied().filter(|&ct| match ct.literal {
                        ConstantKind::Ty(_) => {
                            bug!("should never encounter ty::UnevaluatedConst in `required_consts`")
                        }
                        ConstantKind::Val(..) | ConstantKind::Unevaluated(..) => true,
                    }),
                );
            }
            kind => bug!("unexpected terminator kind {:?}", kind),
        }
    }

    fn make_call_args(
        &self,
        args: Vec<Operand<'tcx>>,
        callsite: &CallSite<'tcx>,
        caller_body: &mut Body<'tcx>,
        callee_body: &Body<'tcx>,
    ) -> Vec<Local> {
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
        if callsite.fn_sig.abi() == Abi::RustCall && callee_body.spread_arg.is_none() {
            let mut args = args.into_iter();
            let self_ = self.create_temp_if_necessary(args.next().unwrap(), callsite, caller_body);
            let tuple = self.create_temp_if_necessary(args.next().unwrap(), callsite, caller_body);
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
                let tuple_field = Operand::Move(tcx.mk_place_field(tuple, Field::new(i), ty));

                // Spill to a local to make e.g., `tmp0`.
                self.create_temp_if_necessary(tuple_field, callsite, caller_body)
            });

            closure_ref_arg.chain(tuple_tmp_args).collect()
        } else {
            args.into_iter()
                .map(|a| self.create_temp_if_necessary(a, callsite, caller_body))
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
        let local = self.new_call_temp(caller_body, callsite, arg_ty);
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
    ) -> Local {
        let local = caller_body.local_decls.push(LocalDecl::new(ty, callsite.source_info.span));

        caller_body[callsite.block].statements.push(Statement {
            source_info: callsite.source_info,
            kind: StatementKind::StorageLive(local),
        });

        if let Some(block) = callsite.target {
            caller_body[block].statements.insert(
                0,
                Statement {
                    source_info: callsite.source_info,
                    kind: StatementKind::StorageDead(local),
                },
            );
        }

        local
    }
}

fn type_size_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    ty: Ty<'tcx>,
) -> Option<u64> {
    tcx.layout_of(param_env.and(ty)).ok().map(|layout| layout.size.bytes())
}

/// Verify that the callee body is compatible with the caller.
///
/// This visitor mostly computes the inlining cost,
/// but also needs to verify that types match because of normalization failure.
struct CostChecker<'b, 'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    cost: usize,
    callee_body: &'b Body<'tcx>,
    instance: ty::Instance<'tcx>,
    validation: Result<(), &'static str>,
}

impl<'tcx> Visitor<'tcx> for CostChecker<'_, 'tcx> {
    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        // Don't count StorageLive/StorageDead in the inlining cost.
        match statement.kind {
            StatementKind::StorageLive(_)
            | StatementKind::StorageDead(_)
            | StatementKind::Deinit(_)
            | StatementKind::Nop => {}
            _ => self.cost += INSTR_COST,
        }

        self.super_statement(statement, location);
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        let tcx = self.tcx;
        match terminator.kind {
            TerminatorKind::Drop { ref place, unwind, .. } => {
                // If the place doesn't actually need dropping, treat it like a regular goto.
                let ty = self.instance.subst_mir(tcx, &place.ty(self.callee_body, tcx).ty);
                if ty.needs_drop(tcx, self.param_env) {
                    self.cost += CALL_PENALTY;
                    if unwind.is_some() {
                        self.cost += LANDINGPAD_PENALTY;
                    }
                } else {
                    self.cost += INSTR_COST;
                }
            }
            TerminatorKind::Call { func: Operand::Constant(ref f), cleanup, .. } => {
                let fn_ty = self.instance.subst_mir(tcx, &f.literal.ty());
                self.cost += if let ty::FnDef(def_id, _) = *fn_ty.kind() && tcx.is_intrinsic(def_id) {
                    // Don't give intrinsics the extra penalty for calls
                    INSTR_COST
                } else {
                    CALL_PENALTY
                };
                if cleanup.is_some() {
                    self.cost += LANDINGPAD_PENALTY;
                }
            }
            TerminatorKind::Assert { cleanup, .. } => {
                self.cost += CALL_PENALTY;
                if cleanup.is_some() {
                    self.cost += LANDINGPAD_PENALTY;
                }
            }
            TerminatorKind::Resume => self.cost += RESUME_PENALTY,
            TerminatorKind::InlineAsm { cleanup, .. } => {
                self.cost += INSTR_COST;
                if cleanup.is_some() {
                    self.cost += LANDINGPAD_PENALTY;
                }
            }
            _ => self.cost += INSTR_COST,
        }

        self.super_terminator(terminator, location);
    }

    /// Count up the cost of local variables and temps, if we know the size
    /// use that, otherwise we use a moderately-large dummy cost.
    fn visit_local_decl(&mut self, local: Local, local_decl: &LocalDecl<'tcx>) {
        let tcx = self.tcx;
        let ptr_size = tcx.data_layout.pointer_size.bytes();

        let ty = self.instance.subst_mir(tcx, &local_decl.ty);
        // Cost of the var is the size in machine-words, if we know
        // it.
        if let Some(size) = type_size_of(tcx, self.param_env, ty) {
            self.cost += ((size + ptr_size - 1) / ptr_size) as usize;
        } else {
            self.cost += UNKNOWN_SIZE_COST;
        }

        self.super_local_decl(local, local_decl)
    }

    /// This method duplicates code from MIR validation in an attempt to detect type mismatches due
    /// to normalization failure.
    fn visit_projection_elem(
        &mut self,
        local: Local,
        proj_base: &[PlaceElem<'tcx>],
        elem: PlaceElem<'tcx>,
        context: PlaceContext,
        location: Location,
    ) {
        if let ProjectionElem::Field(f, ty) = elem {
            let parent = Place { local, projection: self.tcx.mk_place_elems(proj_base) };
            let parent_ty = parent.ty(&self.callee_body.local_decls, self.tcx);
            let check_equal = |this: &mut Self, f_ty| {
                if !util::is_equal_up_to_subtyping(this.tcx, this.param_env, ty, f_ty) {
                    trace!(?ty, ?f_ty);
                    this.validation = Err("failed to normalize projection type");
                    return;
                }
            };

            let kind = match parent_ty.ty.kind() {
                &ty::Alias(ty::Opaque, ty::AliasTy { def_id, substs, .. }) => {
                    self.tcx.type_of(def_id).subst(self.tcx, substs).kind()
                }
                kind => kind,
            };

            match kind {
                ty::Tuple(fields) => {
                    let Some(f_ty) = fields.get(f.as_usize()) else {
                        self.validation = Err("malformed MIR");
                        return;
                    };
                    check_equal(self, *f_ty);
                }
                ty::Adt(adt_def, substs) => {
                    let var = parent_ty.variant_index.unwrap_or(VariantIdx::from_u32(0));
                    let Some(field) = adt_def.variant(var).fields.get(f.as_usize()) else {
                        self.validation = Err("malformed MIR");
                        return;
                    };
                    check_equal(self, field.ty(self.tcx, substs));
                }
                ty::Closure(_, substs) => {
                    let substs = substs.as_closure();
                    let Some(f_ty) = substs.upvar_tys().nth(f.as_usize()) else {
                        self.validation = Err("malformed MIR");
                        return;
                    };
                    check_equal(self, f_ty);
                }
                &ty::Generator(def_id, substs, _) => {
                    let f_ty = if let Some(var) = parent_ty.variant_index {
                        let gen_body = if def_id == self.callee_body.source.def_id() {
                            self.callee_body
                        } else {
                            self.tcx.optimized_mir(def_id)
                        };

                        let Some(layout) = gen_body.generator_layout() else {
                            self.validation = Err("malformed MIR");
                            return;
                        };

                        let Some(&local) = layout.variant_fields[var].get(f) else {
                            self.validation = Err("malformed MIR");
                            return;
                        };

                        let Some(f_ty) = layout.field_tys.get(local) else {
                            self.validation = Err("malformed MIR");
                            return;
                        };

                        f_ty.ty
                    } else {
                        let Some(f_ty) = substs.as_generator().prefix_tys().nth(f.index()) else {
                            self.validation = Err("malformed MIR");
                            return;
                        };

                        f_ty
                    };

                    check_equal(self, f_ty);
                }
                _ => self.validation = Err("malformed MIR"),
            }
        }

        self.super_projection_elem(local, proj_base, elem, context, location);
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
    cleanup_block: Option<BasicBlock>,
    in_cleanup_block: bool,
    tcx: TyCtxt<'tcx>,
    expn_data: LocalExpnId,
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

    fn map_unwind(&self, unwind: Option<BasicBlock>) -> Option<BasicBlock> {
        if self.in_cleanup_block {
            if unwind.is_some() {
                bug!("cleanup on cleanup block");
            }
            return unwind;
        }

        match unwind {
            Some(target) => Some(self.map_block(target)),
            // Add an unwind edge to the original call's cleanup block
            None => self.cleanup_block,
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

    fn visit_span(&mut self, span: &mut Span) {
        // Make sure that all spans track the fact that they were inlined.
        *span = span.fresh_expansion(self.expn_data);
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
            TerminatorKind::GeneratorDrop | TerminatorKind::Yield { .. } => bug!(),
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
            TerminatorKind::Call { ref mut target, ref mut cleanup, .. } => {
                if let Some(ref mut tgt) = *target {
                    *tgt = self.map_block(*tgt);
                }
                *cleanup = self.map_unwind(*cleanup);
            }
            TerminatorKind::Assert { ref mut target, ref mut cleanup, .. } => {
                *target = self.map_block(*target);
                *cleanup = self.map_unwind(*cleanup);
            }
            TerminatorKind::Return => {
                terminator.kind = if let Some(tgt) = self.callsite.target {
                    TerminatorKind::Goto { target: tgt }
                } else {
                    TerminatorKind::Unreachable
                }
            }
            TerminatorKind::Resume => {
                if let Some(tgt) = self.cleanup_block {
                    terminator.kind = TerminatorKind::Goto { target: tgt }
                }
            }
            TerminatorKind::Abort => {}
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
            TerminatorKind::InlineAsm { ref mut destination, ref mut cleanup, .. } => {
                if let Some(ref mut tgt) = *destination {
                    *tgt = self.map_block(*tgt);
                }
                *cleanup = self.map_unwind(*cleanup);
            }
        }
    }
}
