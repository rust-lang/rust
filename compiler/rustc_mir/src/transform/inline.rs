//! Inlining pass for MIR functions

use rustc_attr as attr;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::Idx;
use rustc_middle::middle::codegen_fn_attrs::{CodegenFnAttrFlags, CodegenFnAttrs};
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::subst::Subst;
use rustc_middle::ty::{self, ConstKind, Instance, InstanceDef, ParamEnv, Ty, TyCtxt};
use rustc_span::{hygiene::ExpnKind, ExpnData, Span};
use rustc_target::spec::abi::Abi;

use super::simplify::{remove_dead_blocks, CfgSimplifier};
use crate::transform::MirPass;
use std::collections::VecDeque;
use std::iter;
use std::ops::RangeFrom;

const DEFAULT_THRESHOLD: usize = 50;
const HINT_THRESHOLD: usize = 100;

const INSTR_COST: usize = 5;
const CALL_PENALTY: usize = 25;
const LANDINGPAD_PENALTY: usize = 50;
const RESUME_PENALTY: usize = 45;

const UNKNOWN_SIZE_COST: usize = 10;

pub struct Inline;

#[derive(Copy, Clone, Debug)]
struct CallSite<'tcx> {
    callee: Instance<'tcx>,
    bb: BasicBlock,
    source_info: SourceInfo,
}

impl<'tcx> MirPass<'tcx> for Inline {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if tcx.sess.opts.debugging_opts.mir_opt_level >= 2 {
            if tcx.sess.opts.debugging_opts.instrument_coverage {
                // The current implementation of source code coverage injects code region counters
                // into the MIR, and assumes a 1-to-1 correspondence between MIR and source-code-
                // based function.
                debug!("function inlining is disabled when compiling with `instrument_coverage`");
            } else {
                Inliner {
                    tcx,
                    param_env: tcx.param_env_reveal_all_normalized(body.source.def_id()),
                    codegen_fn_attrs: tcx.codegen_fn_attrs(body.source.def_id()),
                }
                .run_pass(body);
            }
        }
    }
}

struct Inliner<'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    codegen_fn_attrs: &'tcx CodegenFnAttrs,
}

impl Inliner<'tcx> {
    fn run_pass(&self, caller_body: &mut Body<'tcx>) {
        // Keep a queue of callsites to try inlining on. We take
        // advantage of the fact that queries detect cycles here to
        // allow us to try and fetch the fully optimized MIR of a
        // call; if it succeeds, we can inline it and we know that
        // they do not call us.  Otherwise, we just don't try to
        // inline.
        //
        // We use a queue so that we inline "broadly" before we inline
        // in depth. It is unclear if this is the best heuristic,
        // really, but that's true of all the heuristics in this
        // file. =)

        let mut callsites = VecDeque::new();

        let def_id = caller_body.source.def_id();

        // Only do inlining into fn bodies.
        let self_hir_id = self.tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
        if self.tcx.hir().body_owner_kind(self_hir_id).is_fn_or_closure()
            && caller_body.source.promoted.is_none()
        {
            for (bb, bb_data) in caller_body.basic_blocks().iter_enumerated() {
                if let Some(callsite) = self.get_valid_function_call(bb, bb_data, caller_body) {
                    callsites.push_back(callsite);
                }
            }
        } else {
            return;
        }

        let mut changed = false;
        while let Some(callsite) = callsites.pop_front() {
            debug!("checking whether to inline callsite {:?}", callsite);

            if let InstanceDef::Item(_) = callsite.callee.def {
                if !self.tcx.is_mir_available(callsite.callee.def_id()) {
                    debug!("checking whether to inline callsite {:?} - MIR unavailable", callsite,);
                    continue;
                }
            }

            let callee_body = if let Some(callee_def_id) = callsite.callee.def_id().as_local() {
                let callee_hir_id = self.tcx.hir().local_def_id_to_hir_id(callee_def_id);
                // Avoid a cycle here by only using `instance_mir` only if we have
                // a lower `HirId` than the callee. This ensures that the callee will
                // not inline us. This trick only works without incremental compilation.
                // So don't do it if that is enabled. Also avoid inlining into generators,
                // since their `optimized_mir` is used for layout computation, which can
                // create a cycle, even when no attempt is made to inline the function
                // in the other direction.
                if !self.tcx.dep_graph.is_fully_enabled()
                    && self_hir_id < callee_hir_id
                    && caller_body.generator_kind.is_none()
                {
                    self.tcx.instance_mir(callsite.callee.def)
                } else {
                    continue;
                }
            } else {
                // This cannot result in a cycle since the callee MIR is from another crate
                // and is already optimized.
                self.tcx.instance_mir(callsite.callee.def)
            };

            let callee_body: &Body<'tcx> = &*callee_body;

            let callee_body = if self.consider_optimizing(callsite, callee_body) {
                self.tcx.subst_and_normalize_erasing_regions(
                    &callsite.callee.substs,
                    self.param_env,
                    callee_body,
                )
            } else {
                continue;
            };

            // Copy only unevaluated constants from the callee_body into the caller_body.
            // Although we are only pushing `ConstKind::Unevaluated` consts to
            // `required_consts`, here we may not only have `ConstKind::Unevaluated`
            // because we are calling `subst_and_normalize_erasing_regions`.
            caller_body.required_consts.extend(callee_body.required_consts.iter().copied().filter(
                |&constant| matches!(constant.literal.val, ConstKind::Unevaluated(_, _, _)),
            ));

            let start = caller_body.basic_blocks().len();
            debug!("attempting to inline callsite {:?} - body={:?}", callsite, callee_body);
            if !self.inline_call(callsite, caller_body, callee_body) {
                debug!("attempting to inline callsite {:?} - failure", callsite);
                continue;
            }
            debug!("attempting to inline callsite {:?} - success", callsite);

            // Add callsites from inlined function
            for (bb, bb_data) in caller_body.basic_blocks().iter_enumerated().skip(start) {
                if let Some(new_callsite) = self.get_valid_function_call(bb, bb_data, caller_body) {
                    // Don't inline the same function multiple times.
                    if callsite.callee != new_callsite.callee {
                        callsites.push_back(new_callsite);
                    }
                }
            }

            changed = true;
        }

        // Simplify if we inlined anything.
        if changed {
            debug!("running simplify cfg on {:?}", caller_body.source);
            CfgSimplifier::new(caller_body).simplify();
            remove_dead_blocks(caller_body);
        }
    }

    fn get_valid_function_call(
        &self,
        bb: BasicBlock,
        bb_data: &BasicBlockData<'tcx>,
        caller_body: &Body<'tcx>,
    ) -> Option<CallSite<'tcx>> {
        // Don't inline calls that are in cleanup blocks.
        if bb_data.is_cleanup {
            return None;
        }

        // Only consider direct calls to functions
        let terminator = bb_data.terminator();
        if let TerminatorKind::Call { func: ref op, .. } = terminator.kind {
            if let ty::FnDef(callee_def_id, substs) = *op.ty(caller_body, self.tcx).kind() {
                // To resolve an instance its substs have to be fully normalized, so
                // we do this here.
                let normalized_substs = self.tcx.normalize_erasing_regions(self.param_env, substs);
                let callee =
                    Instance::resolve(self.tcx, self.param_env, callee_def_id, normalized_substs)
                        .ok()
                        .flatten()?;

                if let InstanceDef::Virtual(..) | InstanceDef::Intrinsic(_) = callee.def {
                    return None;
                }

                return Some(CallSite { callee, bb, source_info: terminator.source_info });
            }
        }

        None
    }

    fn consider_optimizing(&self, callsite: CallSite<'tcx>, callee_body: &Body<'tcx>) -> bool {
        debug!("consider_optimizing({:?})", callsite);
        self.should_inline(callsite, callee_body)
            && self.tcx.consider_optimizing(|| {
                format!("Inline {:?} into {:?}", callee_body.span, callsite)
            })
    }

    fn should_inline(&self, callsite: CallSite<'tcx>, callee_body: &Body<'tcx>) -> bool {
        debug!("should_inline({:?})", callsite);
        let tcx = self.tcx;

        // Cannot inline generators which haven't been transformed yet
        if callee_body.yield_ty.is_some() {
            debug!("    yield ty present - not inlining");
            return false;
        }

        let codegen_fn_attrs = tcx.codegen_fn_attrs(callsite.callee.def_id());

        let self_features = &self.codegen_fn_attrs.target_features;
        let callee_features = &codegen_fn_attrs.target_features;
        if callee_features.iter().any(|feature| !self_features.contains(feature)) {
            debug!("`callee has extra target features - not inlining");
            return false;
        }

        let self_no_sanitize =
            self.codegen_fn_attrs.no_sanitize & self.tcx.sess.opts.debugging_opts.sanitizer;
        let callee_no_sanitize =
            codegen_fn_attrs.no_sanitize & self.tcx.sess.opts.debugging_opts.sanitizer;
        if self_no_sanitize != callee_no_sanitize {
            debug!("`callee has incompatible no_sanitize attribute - not inlining");
            return false;
        }

        let hinted = match codegen_fn_attrs.inline {
            // Just treat inline(always) as a hint for now,
            // there are cases that prevent inlining that we
            // need to check for first.
            attr::InlineAttr::Always => true,
            attr::InlineAttr::Never => {
                debug!("`#[inline(never)]` present - not inlining");
                return false;
            }
            attr::InlineAttr::Hint => true,
            attr::InlineAttr::None => false,
        };

        // Only inline local functions if they would be eligible for cross-crate
        // inlining. This is to ensure that the final crate doesn't have MIR that
        // reference unexported symbols
        if callsite.callee.def_id().is_local() {
            if callsite.callee.substs.non_erasable_generics().count() == 0 && !hinted {
                debug!("    callee is an exported function - not inlining");
                return false;
            }
        }

        let mut threshold = if hinted { HINT_THRESHOLD } else { DEFAULT_THRESHOLD };

        // Significantly lower the threshold for inlining cold functions
        if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::COLD) {
            threshold /= 5;
        }

        // Give a bonus functions with a small number of blocks,
        // We normally have two or three blocks for even
        // very small functions.
        if callee_body.basic_blocks().len() <= 3 {
            threshold += threshold / 4;
        }
        debug!("    final inline threshold = {}", threshold);

        // FIXME: Give a bonus to functions with only a single caller
        let mut first_block = true;
        let mut cost = 0;

        // Traverse the MIR manually so we can account for the effects of
        // inlining on the CFG.
        let mut work_list = vec![START_BLOCK];
        let mut visited = BitSet::new_empty(callee_body.basic_blocks().len());
        while let Some(bb) = work_list.pop() {
            if !visited.insert(bb.index()) {
                continue;
            }
            let blk = &callee_body.basic_blocks()[bb];

            for stmt in &blk.statements {
                // Don't count StorageLive/StorageDead in the inlining cost.
                match stmt.kind {
                    StatementKind::StorageLive(_)
                    | StatementKind::StorageDead(_)
                    | StatementKind::Nop => {}
                    _ => cost += INSTR_COST,
                }
            }
            let term = blk.terminator();
            let mut is_drop = false;
            match term.kind {
                TerminatorKind::Drop { ref place, target, unwind }
                | TerminatorKind::DropAndReplace { ref place, target, unwind, .. } => {
                    is_drop = true;
                    work_list.push(target);
                    // If the place doesn't actually need dropping, treat it like
                    // a regular goto.
                    let ty = place.ty(callee_body, tcx).subst(tcx, callsite.callee.substs).ty;
                    if ty.needs_drop(tcx, self.param_env) {
                        cost += CALL_PENALTY;
                        if let Some(unwind) = unwind {
                            cost += LANDINGPAD_PENALTY;
                            work_list.push(unwind);
                        }
                    } else {
                        cost += INSTR_COST;
                    }
                }

                TerminatorKind::Unreachable | TerminatorKind::Call { destination: None, .. }
                    if first_block =>
                {
                    // If the function always diverges, don't inline
                    // unless the cost is zero
                    threshold = 0;
                }

                TerminatorKind::Call { func: Operand::Constant(ref f), cleanup, .. } => {
                    if let ty::FnDef(def_id, _) = *f.literal.ty.kind() {
                        // Don't give intrinsics the extra penalty for calls
                        let f = tcx.fn_sig(def_id);
                        if f.abi() == Abi::RustIntrinsic || f.abi() == Abi::PlatformIntrinsic {
                            cost += INSTR_COST;
                        } else {
                            cost += CALL_PENALTY;
                        }
                    } else {
                        cost += CALL_PENALTY;
                    }
                    if cleanup.is_some() {
                        cost += LANDINGPAD_PENALTY;
                    }
                }
                TerminatorKind::Assert { cleanup, .. } => {
                    cost += CALL_PENALTY;

                    if cleanup.is_some() {
                        cost += LANDINGPAD_PENALTY;
                    }
                }
                TerminatorKind::Resume => cost += RESUME_PENALTY,
                _ => cost += INSTR_COST,
            }

            if !is_drop {
                for &succ in term.successors() {
                    work_list.push(succ);
                }
            }

            first_block = false;
        }

        // Count up the cost of local variables and temps, if we know the size
        // use that, otherwise we use a moderately-large dummy cost.

        let ptr_size = tcx.data_layout.pointer_size.bytes();

        for v in callee_body.vars_and_temps_iter() {
            let v = &callee_body.local_decls[v];
            let ty = v.ty.subst(tcx, callsite.callee.substs);
            // Cost of the var is the size in machine-words, if we know
            // it.
            if let Some(size) = type_size_of(tcx, self.param_env, ty) {
                cost += (size / ptr_size) as usize;
            } else {
                cost += UNKNOWN_SIZE_COST;
            }
        }

        if let attr::InlineAttr::Always = codegen_fn_attrs.inline {
            debug!("INLINING {:?} because inline(always) [cost={}]", callsite, cost);
            true
        } else {
            if cost <= threshold {
                debug!("INLINING {:?} [cost={} <= threshold={}]", callsite, cost, threshold);
                true
            } else {
                debug!("NOT inlining {:?} [cost={} > threshold={}]", callsite, cost, threshold);
                false
            }
        }
    }

    fn inline_call(
        &self,
        callsite: CallSite<'tcx>,
        caller_body: &mut Body<'tcx>,
        mut callee_body: Body<'tcx>,
    ) -> bool {
        let terminator = caller_body[callsite.bb].terminator.take().unwrap();
        match terminator.kind {
            // FIXME: Handle inlining of diverging calls
            TerminatorKind::Call { args, destination: Some(destination), cleanup, .. } => {
                debug!("inlined {:?} into {:?}", callsite.callee, caller_body.source);

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

                let dest = if dest_needs_borrow(destination.0) {
                    debug!("creating temp for return destination");
                    let dest = Rvalue::Ref(
                        self.tcx.lifetimes.re_erased,
                        BorrowKind::Mut { allow_two_phase_borrow: false },
                        destination.0,
                    );

                    let ty = dest.ty(caller_body, self.tcx);

                    let temp = LocalDecl::new(ty, callsite.source_info.span);

                    let tmp = caller_body.local_decls.push(temp);
                    let tmp = Place::from(tmp);

                    let stmt = Statement {
                        source_info: callsite.source_info,
                        kind: StatementKind::Assign(box (tmp, dest)),
                    };
                    caller_body[callsite.bb].statements.push(stmt);
                    self.tcx.mk_place_deref(tmp)
                } else {
                    destination.0
                };

                let return_block = destination.1;

                // Copy the arguments if needed.
                let args: Vec<_> = self.make_call_args(args, &callsite, caller_body, return_block);

                let mut integrator = Integrator {
                    args: &args,
                    new_locals: Local::new(caller_body.local_decls.len())..,
                    new_scopes: SourceScope::new(caller_body.source_scopes.len())..,
                    new_blocks: BasicBlock::new(caller_body.basic_blocks().len())..,
                    destination: dest,
                    return_block,
                    cleanup_block: cleanup,
                    in_cleanup_block: false,
                    tcx: self.tcx,
                    callsite_span: callsite.source_info.span,
                    body_span: callee_body.span,
                };

                // Map all `Local`s, `SourceScope`s and `BasicBlock`s to new ones
                // (or existing ones, in a few special cases) in the caller.
                integrator.visit_body(&mut callee_body);

                for scope in &mut callee_body.source_scopes {
                    // FIXME(eddyb) move this into a `fn visit_scope_data` in `Integrator`.
                    if scope.parent_scope.is_none() {
                        let callsite_scope = &caller_body.source_scopes[callsite.source_info.scope];

                        // Attach the outermost callee scope as a child of the callsite
                        // scope, via the `parent_scope` and `inlined_parent_scope` chains.
                        scope.parent_scope = Some(callsite.source_info.scope);
                        assert_eq!(scope.inlined_parent_scope, None);
                        scope.inlined_parent_scope = if callsite_scope.inlined.is_some() {
                            Some(callsite.source_info.scope)
                        } else {
                            callsite_scope.inlined_parent_scope
                        };

                        // Mark the outermost callee scope as an inlined one.
                        assert_eq!(scope.inlined, None);
                        scope.inlined = Some((callsite.callee, callsite.source_info.span));
                    } else if scope.inlined_parent_scope.is_none() {
                        // Make it easy to find the scope with `inlined` set above.
                        scope.inlined_parent_scope =
                            Some(integrator.map_scope(OUTERMOST_SOURCE_SCOPE));
                    }
                }

                // Insert all of the (mapped) parts of the callee body into the caller.
                caller_body.local_decls.extend(
                    // FIXME(eddyb) make `Range<Local>` iterable so that we can use
                    // `callee_body.local_decls.drain(callee_body.vars_and_temps())`
                    callee_body
                        .vars_and_temps_iter()
                        .map(|local| callee_body.local_decls[local].clone()),
                );
                caller_body.source_scopes.extend(callee_body.source_scopes.drain(..));
                caller_body.var_debug_info.extend(callee_body.var_debug_info.drain(..));
                caller_body.basic_blocks_mut().extend(callee_body.basic_blocks_mut().drain(..));

                caller_body[callsite.bb].terminator = Some(Terminator {
                    source_info: callsite.source_info,
                    kind: TerminatorKind::Goto { target: integrator.map_block(START_BLOCK) },
                });

                true
            }
            kind => {
                caller_body[callsite.bb].terminator =
                    Some(Terminator { source_info: terminator.source_info, kind });
                false
            }
        }
    }

    fn make_call_args(
        &self,
        args: Vec<Operand<'tcx>>,
        callsite: &CallSite<'tcx>,
        caller_body: &mut Body<'tcx>,
        return_block: BasicBlock,
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
        // FIXME(eddyb) make this check for `"rust-call"` ABI combined with
        // `callee_body.spread_arg == None`, instead of special-casing closures.
        if tcx.is_closure(callsite.callee.def_id()) {
            let mut args = args.into_iter();
            let self_ = self.create_temp_if_necessary(
                args.next().unwrap(),
                callsite,
                caller_body,
                return_block,
            );
            let tuple = self.create_temp_if_necessary(
                args.next().unwrap(),
                callsite,
                caller_body,
                return_block,
            );
            assert!(args.next().is_none());

            let tuple = Place::from(tuple);
            let tuple_tys = if let ty::Tuple(s) = tuple.ty(caller_body, tcx).ty.kind() {
                s
            } else {
                bug!("Closure arguments are not passed as a tuple");
            };

            // The `closure_ref` in our example above.
            let closure_ref_arg = iter::once(self_);

            // The `tmp0`, `tmp1`, and `tmp2` in our example abonve.
            let tuple_tmp_args = tuple_tys.iter().enumerate().map(|(i, ty)| {
                // This is e.g., `tuple_tmp.0` in our example above.
                let tuple_field =
                    Operand::Move(tcx.mk_place_field(tuple, Field::new(i), ty.expect_ty()));

                // Spill to a local to make e.g., `tmp0`.
                self.create_temp_if_necessary(tuple_field, callsite, caller_body, return_block)
            });

            closure_ref_arg.chain(tuple_tmp_args).collect()
        } else {
            args.into_iter()
                .map(|a| self.create_temp_if_necessary(a, callsite, caller_body, return_block))
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
        return_block: BasicBlock,
    ) -> Local {
        // FIXME: Analysis of the usage of the arguments to avoid
        // unnecessary temporaries.

        if let Operand::Move(place) = &arg {
            if let Some(local) = place.as_local() {
                if caller_body.local_kind(local) == LocalKind::Temp {
                    // Reuse the operand if it's a temporary already
                    return local;
                }
            }
        }

        debug!("creating temp for argument {:?}", arg);
        // Otherwise, create a temporary for the arg
        let arg = Rvalue::Use(arg);

        let ty = arg.ty(caller_body, self.tcx);

        let arg_tmp = LocalDecl::new(ty, callsite.source_info.span);
        let arg_tmp = caller_body.local_decls.push(arg_tmp);

        caller_body[callsite.bb].statements.push(Statement {
            source_info: callsite.source_info,
            kind: StatementKind::StorageLive(arg_tmp),
        });
        caller_body[callsite.bb].statements.push(Statement {
            source_info: callsite.source_info,
            kind: StatementKind::Assign(box (Place::from(arg_tmp), arg)),
        });
        caller_body[return_block].statements.insert(
            0,
            Statement {
                source_info: callsite.source_info,
                kind: StatementKind::StorageDead(arg_tmp),
            },
        );

        arg_tmp
    }
}

fn type_size_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    ty: Ty<'tcx>,
) -> Option<u64> {
    tcx.layout_of(param_env.and(ty)).ok().map(|layout| layout.size.bytes())
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
    destination: Place<'tcx>,
    return_block: BasicBlock,
    cleanup_block: Option<BasicBlock>,
    in_cleanup_block: bool,
    tcx: TyCtxt<'tcx>,
    callsite_span: Span,
    body_span: Span,
}

impl<'a, 'tcx> Integrator<'a, 'tcx> {
    fn map_local(&self, local: Local) -> Local {
        let new = if local == RETURN_PLACE {
            self.destination.local
        } else {
            let idx = local.index() - 1;
            if idx < self.args.len() {
                self.args[idx]
            } else {
                Local::new(self.new_locals.start.index() + (idx - self.args.len()))
            }
        };
        debug!("mapping local `{:?}` to `{:?}`", local, new);
        new
    }

    fn map_scope(&self, scope: SourceScope) -> SourceScope {
        let new = SourceScope::new(self.new_scopes.start.index() + scope.index());
        debug!("mapping scope `{:?}` to `{:?}`", scope, new);
        new
    }

    fn map_block(&self, block: BasicBlock) -> BasicBlock {
        let new = BasicBlock::new(self.new_blocks.start.index() + block.index());
        debug!("mapping block `{:?}` to `{:?}`", block, new);
        new
    }
}

impl<'a, 'tcx> MutVisitor<'tcx> for Integrator<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_local(&mut self, local: &mut Local, _ctxt: PlaceContext, _location: Location) {
        *local = self.map_local(*local);
    }

    fn visit_source_scope(&mut self, scope: &mut SourceScope) {
        *scope = self.map_scope(*scope);
    }

    fn visit_span(&mut self, span: &mut Span) {
        // Make sure that all spans track the fact that they were inlined.
        *span = self.callsite_span.fresh_expansion(ExpnData {
            def_site: self.body_span,
            ..ExpnData::default(ExpnKind::Inlined, *span, self.tcx.sess.edition(), None)
        });
    }

    fn visit_place(&mut self, place: &mut Place<'tcx>, context: PlaceContext, location: Location) {
        // If this is the `RETURN_PLACE`, we need to rebase any projections onto it.
        let dest_proj_len = self.destination.projection.len();
        if place.local == RETURN_PLACE && dest_proj_len > 0 {
            let mut projs = Vec::with_capacity(dest_proj_len + place.projection.len());
            projs.extend(self.destination.projection);
            projs.extend(place.projection);

            place.projection = self.tcx.intern_place_elems(&*projs);
        }
        // Handles integrating any locals that occur in the base
        // or projections
        self.super_place(place, context, location)
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
            TerminatorKind::Drop { ref mut target, ref mut unwind, .. }
            | TerminatorKind::DropAndReplace { ref mut target, ref mut unwind, .. } => {
                *target = self.map_block(*target);
                if let Some(tgt) = *unwind {
                    *unwind = Some(self.map_block(tgt));
                } else if !self.in_cleanup_block {
                    // Unless this drop is in a cleanup block, add an unwind edge to
                    // the original call's cleanup block
                    *unwind = self.cleanup_block;
                }
            }
            TerminatorKind::Call { ref mut destination, ref mut cleanup, .. } => {
                if let Some((_, ref mut tgt)) = *destination {
                    *tgt = self.map_block(*tgt);
                }
                if let Some(tgt) = *cleanup {
                    *cleanup = Some(self.map_block(tgt));
                } else if !self.in_cleanup_block {
                    // Unless this call is in a cleanup block, add an unwind edge to
                    // the original call's cleanup block
                    *cleanup = self.cleanup_block;
                }
            }
            TerminatorKind::Assert { ref mut target, ref mut cleanup, .. } => {
                *target = self.map_block(*target);
                if let Some(tgt) = *cleanup {
                    *cleanup = Some(self.map_block(tgt));
                } else if !self.in_cleanup_block {
                    // Unless this assert is in a cleanup block, add an unwind edge to
                    // the original call's cleanup block
                    *cleanup = self.cleanup_block;
                }
            }
            TerminatorKind::Return => {
                terminator.kind = TerminatorKind::Goto { target: self.return_block };
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
            TerminatorKind::InlineAsm { ref mut destination, .. } => {
                if let Some(ref mut tgt) = *destination {
                    *tgt = self.map_block(*tgt);
                }
            }
        }
    }
}
