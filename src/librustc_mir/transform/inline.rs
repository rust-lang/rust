//! Inlining pass for MIR functions

use rustc::hir::CodegenFnAttrFlags;
use rustc::hir::def_id::DefId;

use rustc_data_structures::bit_set::BitSet;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};

use rustc::mir::*;
use rustc::mir::visit::*;
use rustc::ty::{self, Instance, InstanceDef, ParamEnv, Ty, TyCtxt};
use rustc::ty::subst::{Subst, SubstsRef};

use std::collections::VecDeque;
use std::iter;
use crate::transform::{MirPass, MirSource};
use super::simplify::{remove_dead_blocks, CfgSimplifier};

use syntax::attr;
use rustc_target::spec::abi::Abi;

const DEFAULT_THRESHOLD: usize = 50;
const HINT_THRESHOLD: usize = 100;

const INSTR_COST: usize = 5;
const CALL_PENALTY: usize = 25;

const UNKNOWN_SIZE_COST: usize = 10;

pub struct Inline;

#[derive(Copy, Clone, Debug)]
struct CallSite<'tcx> {
    callee: DefId,
    substs: SubstsRef<'tcx>,
    bb: BasicBlock,
    location: SourceInfo,
}

impl MirPass for Inline {
    fn run_pass<'tcx>(&self, tcx: TyCtxt<'tcx>, source: MirSource<'tcx>, body: &mut Body<'tcx>) {
        if tcx.sess.opts.debugging_opts.mir_opt_level >= 2 {
            Inliner { tcx, source }.run_pass(body);
        }
    }
}

struct Inliner<'tcx> {
    tcx: TyCtxt<'tcx>,
    source: MirSource<'tcx>,
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

        let param_env = self.tcx.param_env(self.source.def_id());

        // Only do inlining into fn bodies.
        let id = self.tcx.hir().as_local_hir_id(self.source.def_id()).unwrap();
        if self.tcx.hir().body_owner_kind(id).is_fn_or_closure()
            && self.source.promoted.is_none()
        {
            for (bb, bb_data) in caller_body.basic_blocks().iter_enumerated() {
                if let Some(callsite) = self.get_valid_function_call(bb,
                                                                    bb_data,
                                                                    caller_body,
                                                                    param_env) {
                    callsites.push_back(callsite);
                }
            }
        } else {
            return;
        }

        let mut local_change;
        let mut changed = false;

        loop {
            local_change = false;
            while let Some(callsite) = callsites.pop_front() {
                debug!("checking whether to inline callsite {:?}", callsite);
                if !self.tcx.is_mir_available(callsite.callee) {
                    debug!("checking whether to inline callsite {:?} - MIR unavailable", callsite);
                    continue;
                }

                let self_node_id = self.tcx.hir().as_local_node_id(self.source.def_id()).unwrap();
                let callee_node_id = self.tcx.hir().as_local_node_id(callsite.callee);

                let callee_body = if let Some(callee_node_id) = callee_node_id {
                    // Avoid a cycle here by only using `optimized_mir` only if we have
                    // a lower node id than the callee. This ensures that the callee will
                    // not inline us. This trick only works without incremental compilation.
                    // So don't do it if that is enabled.
                    if !self.tcx.dep_graph.is_fully_enabled()
                        && self_node_id.as_u32() < callee_node_id.as_u32() {
                        self.tcx.optimized_mir(callsite.callee)
                    } else {
                        continue;
                    }
                } else {
                    // This cannot result in a cycle since the callee MIR is from another crate
                    // and is already optimized.
                    self.tcx.optimized_mir(callsite.callee)
                };

                let callee_body = if self.consider_optimizing(callsite, callee_body) {
                    self.tcx.subst_and_normalize_erasing_regions(
                        &callsite.substs,
                        param_env,
                        callee_body,
                    )
                } else {
                    continue;
                };

                let start = caller_body.basic_blocks().len();
                debug!("attempting to inline callsite {:?} - body={:?}", callsite, callee_body);
                if !self.inline_call(callsite, caller_body, callee_body) {
                    debug!("attempting to inline callsite {:?} - failure", callsite);
                    continue;
                }
                debug!("attempting to inline callsite {:?} - success", callsite);

                // Add callsites from inlined function
                for (bb, bb_data) in caller_body.basic_blocks().iter_enumerated().skip(start) {
                    if let Some(new_callsite) = self.get_valid_function_call(bb,
                                                                             bb_data,
                                                                             caller_body,
                                                                             param_env) {
                        // Don't inline the same function multiple times.
                        if callsite.callee != new_callsite.callee {
                            callsites.push_back(new_callsite);
                        }
                    }
                }

                local_change = true;
                changed = true;
            }

            if !local_change {
                break;
            }
        }

        // Simplify if we inlined anything.
        if changed {
            debug!("Running simplify cfg on {:?}", self.source);
            CfgSimplifier::new(caller_body).simplify();
            remove_dead_blocks(caller_body);
        }
    }

    fn get_valid_function_call(&self,
                               bb: BasicBlock,
                               bb_data: &BasicBlockData<'tcx>,
                               caller_body: &Body<'tcx>,
                               param_env: ParamEnv<'tcx>,
    ) -> Option<CallSite<'tcx>> {
        // Don't inline calls that are in cleanup blocks.
        if bb_data.is_cleanup { return None; }

        // Only consider direct calls to functions
        let terminator = bb_data.terminator();
        if let TerminatorKind::Call { func: ref op, .. } = terminator.kind {
            if let ty::FnDef(callee_def_id, substs) = op.ty(caller_body, self.tcx).sty {
                let instance = Instance::resolve(self.tcx,
                                                 param_env,
                                                 callee_def_id,
                                                 substs)?;

                if let InstanceDef::Virtual(..) = instance.def {
                    return None;
                }

                return Some(CallSite {
                    callee: instance.def_id(),
                    substs: instance.substs,
                    bb,
                    location: terminator.source_info
                });
            }
        }

        None
    }

    fn consider_optimizing(&self,
                           callsite: CallSite<'tcx>,
                           callee_body: &Body<'tcx>)
                           -> bool
    {
        debug!("consider_optimizing({:?})", callsite);
        self.should_inline(callsite, callee_body)
            && self.tcx.consider_optimizing(|| format!("Inline {:?} into {:?}",
                                                       callee_body.span,
                                                       callsite))
    }

    fn should_inline(&self,
                     callsite: CallSite<'tcx>,
                     callee_body: &Body<'tcx>)
                     -> bool
    {
        debug!("should_inline({:?})", callsite);
        let tcx = self.tcx;

        // Don't inline closures that have capture debuginfo
        // FIXME: Handle closures better
        if callee_body.__upvar_debuginfo_codegen_only_do_not_use.len() > 0 {
            debug!("    upvar debuginfo present - not inlining");
            return false;
        }

        // Cannot inline generators which haven't been transformed yet
        if callee_body.yield_ty.is_some() {
            debug!("    yield ty present - not inlining");
            return false;
        }

        // Do not inline {u,i}128 lang items, codegen const eval depends
        // on detecting calls to these lang items and intercepting them
        if tcx.is_binop_lang_item(callsite.callee).is_some() {
            debug!("    not inlining 128bit integer lang item");
            return false;
        }

        let codegen_fn_attrs = tcx.codegen_fn_attrs(callsite.callee);

        let hinted = match codegen_fn_attrs.inline {
            // Just treat inline(always) as a hint for now,
            // there are cases that prevent inlining that we
            // need to check for first.
            attr::InlineAttr::Always => true,
            attr::InlineAttr::Never => {
                debug!("#[inline(never)] present - not inlining");
                return false
            }
            attr::InlineAttr::Hint => true,
            attr::InlineAttr::None => false,
        };

        // Only inline local functions if they would be eligible for cross-crate
        // inlining. This is to ensure that the final crate doesn't have MIR that
        // reference unexported symbols
        if callsite.callee.is_local() {
            if callsite.substs.non_erasable_generics().count() == 0 && !hinted {
                debug!("    callee is an exported function - not inlining");
                return false;
            }
        }

        let mut threshold = if hinted {
            HINT_THRESHOLD
        } else {
            DEFAULT_THRESHOLD
        };

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

        let param_env = tcx.param_env(self.source.def_id());

        let mut first_block = true;
        let mut cost = 0;

        // Traverse the MIR manually so we can account for the effects of
        // inlining on the CFG.
        let mut work_list = vec![START_BLOCK];
        let mut visited = BitSet::new_empty(callee_body.basic_blocks().len());
        while let Some(bb) = work_list.pop() {
            if !visited.insert(bb.index()) { continue; }
            let blk = &callee_body.basic_blocks()[bb];

            for stmt in &blk.statements {
                // Don't count StorageLive/StorageDead in the inlining cost.
                match stmt.kind {
                    StatementKind::StorageLive(_) |
                    StatementKind::StorageDead(_) |
                    StatementKind::Nop => {}
                    _ => cost += INSTR_COST
                }
            }
            let term = blk.terminator();
            let mut is_drop = false;
            match term.kind {
                TerminatorKind::Drop { ref location, target, unwind } |
                TerminatorKind::DropAndReplace { ref location, target, unwind, .. } => {
                    is_drop = true;
                    work_list.push(target);
                    // If the location doesn't actually need dropping, treat it like
                    // a regular goto.
                    let ty = location.ty(callee_body, tcx).subst(tcx, callsite.substs).ty;
                    if ty.needs_drop(tcx, param_env) {
                        cost += CALL_PENALTY;
                        if let Some(unwind) = unwind {
                            work_list.push(unwind);
                        }
                    } else {
                        cost += INSTR_COST;
                    }
                }

                TerminatorKind::Unreachable |
                TerminatorKind::Call { destination: None, .. } if first_block => {
                    // If the function always diverges, don't inline
                    // unless the cost is zero
                    threshold = 0;
                }

                TerminatorKind::Call {func: Operand::Constant(ref f), .. } => {
                    if let ty::FnDef(def_id, _) = f.ty.sty {
                        // Don't give intrinsics the extra penalty for calls
                        let f = tcx.fn_sig(def_id);
                        if f.abi() == Abi::RustIntrinsic || f.abi() == Abi::PlatformIntrinsic {
                            cost += INSTR_COST;
                        } else {
                            cost += CALL_PENALTY;
                        }
                    }
                }
                TerminatorKind::Assert { .. } => cost += CALL_PENALTY,
                _ => cost += INSTR_COST
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
            let ty = v.ty.subst(tcx, callsite.substs);
            // Cost of the var is the size in machine-words, if we know
            // it.
            if let Some(size) = type_size_of(tcx, param_env.clone(), ty) {
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

    fn inline_call(&self,
                   callsite: CallSite<'tcx>,
                   caller_body: &mut Body<'tcx>,
                   mut callee_body: Body<'tcx>) -> bool {
        let terminator = caller_body[callsite.bb].terminator.take().unwrap();
        match terminator.kind {
            // FIXME: Handle inlining of diverging calls
            TerminatorKind::Call { args, destination: Some(destination), cleanup, .. } => {
                debug!("Inlined {:?} into {:?}", callsite.callee, self.source);

                let mut local_map = IndexVec::with_capacity(callee_body.local_decls.len());
                let mut scope_map = IndexVec::with_capacity(callee_body.source_scopes.len());
                let mut promoted_map = IndexVec::with_capacity(callee_body.promoted.len());

                for mut scope in callee_body.source_scopes.iter().cloned() {
                    if scope.parent_scope.is_none() {
                        scope.parent_scope = Some(callsite.location.scope);
                        scope.span = callee_body.span;
                    }

                    scope.span = callsite.location.span;

                    let idx = caller_body.source_scopes.push(scope);
                    scope_map.push(idx);
                }

                for loc in callee_body.vars_and_temps_iter() {
                    let mut local = callee_body.local_decls[loc].clone();

                    local.source_info.scope =
                        scope_map[local.source_info.scope];
                    local.source_info.span = callsite.location.span;
                    local.visibility_scope = scope_map[local.visibility_scope];

                    let idx = caller_body.local_decls.push(local);
                    local_map.push(idx);
                }

                promoted_map.extend(
                    callee_body.promoted.iter().cloned().map(|p| caller_body.promoted.push(p))
                );

                // If the call is something like `a[*i] = f(i)`, where
                // `i : &mut usize`, then just duplicating the `a[*i]`
                // Place could result in two different locations if `f`
                // writes to `i`. To prevent this we need to create a temporary
                // borrow of the place and pass the destination as `*temp` instead.
                fn dest_needs_borrow(place: &Place<'_>) -> bool {
                    place.iterate(|place_base, place_projection| {
                        for proj in place_projection {
                            match proj.elem {
                                ProjectionElem::Deref |
                                ProjectionElem::Index(_) => return true,
                                _ => {}
                            }
                        }

                        match place_base {
                            // Static variables need a borrow because the callee
                            // might modify the same static.
                            PlaceBase::Static(_) => true,
                            _ => false
                        }
                    })
                }

                let dest = if dest_needs_borrow(&destination.0) {
                    debug!("Creating temp for return destination");
                    let dest = Rvalue::Ref(
                        self.tcx.lifetimes.re_erased,
                        BorrowKind::Mut { allow_two_phase_borrow: false },
                        destination.0);

                    let ty = dest.ty(caller_body, self.tcx);

                    let temp = LocalDecl::new_temp(ty, callsite.location.span);

                    let tmp = caller_body.local_decls.push(temp);
                    let tmp = Place::from(tmp);

                    let stmt = Statement {
                        source_info: callsite.location,
                        kind: StatementKind::Assign(tmp.clone(), box dest)
                    };
                    caller_body[callsite.bb]
                        .statements.push(stmt);
                    tmp.deref()
                } else {
                    destination.0
                };

                let return_block = destination.1;

                // Copy the arguments if needed.
                let args: Vec<_> = self.make_call_args(args, &callsite, caller_body);

                let bb_len = caller_body.basic_blocks().len();
                let mut integrator = Integrator {
                    block_idx: bb_len,
                    args: &args,
                    local_map,
                    scope_map,
                    promoted_map,
                    _callsite: callsite,
                    destination: dest,
                    return_block,
                    cleanup_block: cleanup,
                    in_cleanup_block: false
                };


                for (bb, mut block) in callee_body.basic_blocks_mut().drain_enumerated(..) {
                    integrator.visit_basic_block_data(bb, &mut block);
                    caller_body.basic_blocks_mut().push(block);
                }

                let terminator = Terminator {
                    source_info: callsite.location,
                    kind: TerminatorKind::Goto { target: BasicBlock::new(bb_len) }
                };

                caller_body[callsite.bb].terminator = Some(terminator);

                true
            }
            kind => {
                caller_body[callsite.bb].terminator = Some(Terminator {
                    source_info: terminator.source_info,
                    kind,
                });
                false
            }
        }
    }

    fn make_call_args(
        &self,
        args: Vec<Operand<'tcx>>,
        callsite: &CallSite<'tcx>,
        caller_body: &mut Body<'tcx>,
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
        if tcx.is_closure(callsite.callee) {
            let mut args = args.into_iter();
            let self_ = self.create_temp_if_necessary(args.next().unwrap(), callsite, caller_body);
            let tuple = self.create_temp_if_necessary(args.next().unwrap(), callsite, caller_body);
            assert!(args.next().is_none());

            let tuple = Place::from(tuple);
            let tuple_tys = if let ty::Tuple(s) = tuple.ty(caller_body, tcx).ty.sty {
                s
            } else {
                bug!("Closure arguments are not passed as a tuple");
            };

            // The `closure_ref` in our example above.
            let closure_ref_arg = iter::once(self_);

            // The `tmp0`, `tmp1`, and `tmp2` in our example abonve.
            let tuple_tmp_args =
                tuple_tys.iter().enumerate().map(|(i, ty)| {
                    // This is e.g., `tuple_tmp.0` in our example above.
                    let tuple_field = Operand::Move(tuple.clone().field(
                        Field::new(i),
                        ty.expect_ty(),
                    ));

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
        // FIXME: Analysis of the usage of the arguments to avoid
        // unnecessary temporaries.

        if let Operand::Move(Place::Base(PlaceBase::Local(local))) = arg {
            if caller_body.local_kind(local) == LocalKind::Temp {
                // Reuse the operand if it's a temporary already
                return local;
            }
        }

        debug!("Creating temp for argument {:?}", arg);
        // Otherwise, create a temporary for the arg
        let arg = Rvalue::Use(arg);

        let ty = arg.ty(caller_body, self.tcx);

        let arg_tmp = LocalDecl::new_temp(ty, callsite.location.span);
        let arg_tmp = caller_body.local_decls.push(arg_tmp);

        let stmt = Statement {
            source_info: callsite.location,
            kind: StatementKind::Assign(Place::from(arg_tmp), box arg),
        };
        caller_body[callsite.bb].statements.push(stmt);
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
    block_idx: usize,
    args: &'a [Local],
    local_map: IndexVec<Local, Local>,
    scope_map: IndexVec<SourceScope, SourceScope>,
    promoted_map: IndexVec<Promoted, Promoted>,
    _callsite: CallSite<'tcx>,
    destination: Place<'tcx>,
    return_block: BasicBlock,
    cleanup_block: Option<BasicBlock>,
    in_cleanup_block: bool,
}

impl<'a, 'tcx> Integrator<'a, 'tcx> {
    fn update_target(&self, tgt: BasicBlock) -> BasicBlock {
        let new = BasicBlock::new(tgt.index() + self.block_idx);
        debug!("Updating target `{:?}`, new: `{:?}`", tgt, new);
        new
    }
}

impl<'a, 'tcx> MutVisitor<'tcx> for Integrator<'a, 'tcx> {
    fn visit_local(&mut self,
                   local: &mut Local,
                   _ctxt: PlaceContext,
                   _location: Location) {
        if *local == RETURN_PLACE {
            match self.destination {
                Place::Base(PlaceBase::Local(l)) => {
                    *local = l;
                    return;
                },
                ref place => bug!("Return place is {:?}, not local", place)
            }
        }
        let idx = local.index() - 1;
        if idx < self.args.len() {
            *local = self.args[idx];
            return;
        }
        *local = self.local_map[Local::new(idx - self.args.len())];
    }

    fn visit_place(&mut self,
                    place: &mut Place<'tcx>,
                    _ctxt: PlaceContext,
                    _location: Location) {

        match place {
            Place::Base(PlaceBase::Local(RETURN_PLACE)) => {
                // Return pointer; update the place itself
                *place = self.destination.clone();
            },
            Place::Base(
                PlaceBase::Static(box Static { kind: StaticKind::Promoted(promoted), .. })
            ) => {
                if let Some(p) = self.promoted_map.get(*promoted).cloned() {
                    *promoted = p;
                }
            },
            _ => self.super_place(place, _ctxt, _location)
        }
    }

    fn visit_basic_block_data(&mut self, block: BasicBlock, data: &mut BasicBlockData<'tcx>) {
        self.in_cleanup_block = data.is_cleanup;
        self.super_basic_block_data(block, data);
        self.in_cleanup_block = false;
    }

    fn visit_retag(
        &mut self,
        kind: &mut RetagKind,
        place: &mut Place<'tcx>,
        loc: Location,
    ) {
        self.super_retag(kind, place, loc);

        // We have to patch all inlined retags to be aware that they are no longer
        // happening on function entry.
        if *kind == RetagKind::FnEntry {
            *kind = RetagKind::Default;
        }
    }

    fn visit_terminator_kind(&mut self,
                             kind: &mut TerminatorKind<'tcx>, loc: Location) {
        self.super_terminator_kind(kind, loc);

        match *kind {
            TerminatorKind::GeneratorDrop |
            TerminatorKind::Yield { .. } => bug!(),
            TerminatorKind::Goto { ref mut target} => {
                *target = self.update_target(*target);
            }
            TerminatorKind::SwitchInt { ref mut targets, .. } => {
                for tgt in targets {
                    *tgt = self.update_target(*tgt);
                }
            }
            TerminatorKind::Drop { ref mut target, ref mut unwind, .. } |
            TerminatorKind::DropAndReplace { ref mut target, ref mut unwind, .. } => {
                *target = self.update_target(*target);
                if let Some(tgt) = *unwind {
                    *unwind = Some(self.update_target(tgt));
                } else if !self.in_cleanup_block {
                    // Unless this drop is in a cleanup block, add an unwind edge to
                    // the original call's cleanup block
                    *unwind = self.cleanup_block;
                }
            }
            TerminatorKind::Call { ref mut destination, ref mut cleanup, .. } => {
                if let Some((_, ref mut tgt)) = *destination {
                    *tgt = self.update_target(*tgt);
                }
                if let Some(tgt) = *cleanup {
                    *cleanup = Some(self.update_target(tgt));
                } else if !self.in_cleanup_block {
                    // Unless this call is in a cleanup block, add an unwind edge to
                    // the original call's cleanup block
                    *cleanup = self.cleanup_block;
                }
            }
            TerminatorKind::Assert { ref mut target, ref mut cleanup, .. } => {
                *target = self.update_target(*target);
                if let Some(tgt) = *cleanup {
                    *cleanup = Some(self.update_target(tgt));
                } else if !self.in_cleanup_block {
                    // Unless this assert is in a cleanup block, add an unwind edge to
                    // the original call's cleanup block
                    *cleanup = self.cleanup_block;
                }
            }
            TerminatorKind::Return => {
                *kind = TerminatorKind::Goto { target: self.return_block };
            }
            TerminatorKind::Resume => {
                if let Some(tgt) = self.cleanup_block {
                    *kind = TerminatorKind::Goto { target: tgt }
                }
            }
            TerminatorKind::Abort => { }
            TerminatorKind::Unreachable => { }
            TerminatorKind::FalseEdges { ref mut real_target, ref mut imaginary_target } => {
                *real_target = self.update_target(*real_target);
                *imaginary_target = self.update_target(*imaginary_target);
            }
            TerminatorKind::FalseUnwind { real_target: _ , unwind: _ } =>
                // see the ordering of passes in the optimized_mir query.
                bug!("False unwinds should have been removed before inlining")
        }
    }

    fn visit_source_scope(&mut self, scope: &mut SourceScope) {
        *scope = self.scope_map[*scope];
    }
}
