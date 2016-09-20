// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Inlining pass for MIR functions
//!
//! Inlines functions. Is quite conservative in it's conditions, functions
//! that can unwind are not inlined.

use rustc::hir::def_id::DefId;

use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use rustc_data_structures::graph;

use rustc::dep_graph::DepNode;
use rustc::mir::mir_map::MirMap;
use rustc::mir::repr::*;
use rustc::mir::transform::{MirMapPass, MirPassHook, MirSource, Pass};
use rustc::mir::visit::*;
use rustc::traits;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::subst::{Subst,Substs};
use rustc::util::nodemap::{DefIdMap, DefIdSet};

use syntax::attr;
use syntax_pos::Span;

use callgraph;

const DEFAULT_THRESHOLD : usize = 50;
const HINT_THRESHOLD : usize = 100;

const INSTR_COST : usize = 5;
const CALL_PENALTY : usize = 25;

const UNKNOWN_SIZE_COST : usize = 10;

use std::rc::Rc;

pub struct Inline;

impl<'tcx> MirMapPass<'tcx> for Inline {
    fn run_pass<'a>(
        &mut self,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        map: &mut MirMap<'tcx>,
        _: &mut [Box<for<'s> MirPassHook<'s>>]) {

        match tcx.sess.opts.debugging_opts.mir_opt_level {
            Some(0) |
            Some(1) |
            None => { return; },
            _ => {}
        };

        let _ignore = tcx.dep_graph.in_ignore();

        let callgraph = callgraph::CallGraph::build(map);

        let mut inliner = Inliner {
            tcx: tcx,
            foreign_mirs: DefIdMap()
        };

        for scc in callgraph.scc_iter() {
            debug!("Inlining SCC {:?}", scc);
            inliner.inline_scc(map, &callgraph, &scc);
        }
    }
}

impl<'tcx> Pass for Inline { }

struct Inliner<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    foreign_mirs: DefIdMap<Rc<Mir<'tcx>>>,
}

#[derive(Copy, Clone)]
struct CallSite<'tcx> {
    caller: DefId,
    callee: DefId,
    substs: &'tcx Substs<'tcx>,
    bb: BasicBlock,
    location: SourceInfo,
}

impl<'a, 'tcx> Inliner<'a, 'tcx> {
    fn inline_scc(&mut self, map: &mut MirMap<'tcx>,
                            callgraph: &callgraph::CallGraph, scc: &[graph::NodeIndex]) -> bool {
        let mut callsites = Vec::new();
        let mut in_scc = DefIdSet();

        for &node in scc {
            let def_id = callgraph.def_id(node);

            // Don't inspect functions from other crates
            let id = if let Some(id) = self.tcx.map.as_local_node_id(def_id) {
                id
            } else {
                continue;
            };
            let src = MirSource::from_node(self.tcx, id);
            if let MirSource::Fn(_) = src {
                let mir = if let Some(m) = map.map.get(&def_id) {
                    m
                } else {
                    continue;
                };
                for (bb, bb_data) in mir.basic_blocks().iter_enumerated() {
                    // Only consider direct calls to functions
                    let terminator = bb_data.terminator();
                    if let TerminatorKind::Call {
                        func: Operand::Constant(ref f), .. } = terminator.kind {
                        if let ty::TyFnDef(callee_def_id, substs, _) = f.ty.sty {
                            callsites.push(CallSite {
                                caller: def_id,
                                callee: callee_def_id,
                                substs: substs,
                                bb: bb,
                                location: terminator.source_info
                            });
                        }
                    }
                }

                in_scc.insert(def_id);
            }
        }

        // Move callsites that are in the the SCC to the end so
        // they're inlined after calls to outside the SCC
        let mut first_call_in_scc = callsites.len();

        let mut i = 0;
        while i < first_call_in_scc {
            let f = callsites[i].caller;
            if in_scc.contains(&f) {
                first_call_in_scc -= 1;
                callsites.swap(i, first_call_in_scc);
            } else {
                i += 1;
            }
        }

        let mut local_change;
        let mut changed = false;

        loop {
            local_change = false;
            let mut csi = 0;
            while csi < callsites.len() {
                let foreign_mir;

                let callsite = callsites[csi];
                csi += 1;

                let callee_mir = {
                    let callee_mir : Option<&Mir<'tcx>> = if callsite.callee.is_local() {
                        map.map.get(&callsite.callee)
                    } else {
                        foreign_mir = self.get_foreign_mir(callsite.callee);
                        foreign_mir.as_ref().map(|m| &**m)
                    };

                    let callee_mir = if let Some(m) = callee_mir {
                        m
                    } else {
                        continue;
                    };

                    if !self.should_inline(callsite, callee_mir) {
                        continue;
                    }

                    callee_mir.subst(self.tcx, callsite.substs)
                };

                let caller_mir = map.map.get_mut(&callsite.caller).unwrap();

                let start = caller_mir.basic_blocks().len();

                if !self.inline_call(callsite, caller_mir, callee_mir) {
                    continue;
                }

                // Add callsites from inlined function
                for (bb, bb_data) in caller_mir.basic_blocks().iter_enumerated().skip(start) {
                    // Only consider direct calls to functions
                    let terminator = bb_data.terminator();
                    if let TerminatorKind::Call {
                        func: Operand::Constant(ref f), .. } = terminator.kind {
                        if let ty::TyFnDef(callee_def_id, substs, _) = f.ty.sty {
                            // Don't inline the same function multiple times.
                            if callsite.callee != callee_def_id {
                                callsites.push(CallSite {
                                    caller: callsite.caller,
                                    callee: callee_def_id,
                                    substs: substs,
                                    bb: bb,
                                    location: terminator.source_info
                                });
                            }
                        }
                    }
                }


                csi -= 1;
                if scc.len() == 1 {
                    callsites.swap_remove(csi);
                } else {
                    callsites.remove(csi);
                }

                local_change = true;
                changed = true;
            }

            if !local_change {
                break;
            }
        }

        changed
    }

    fn get_foreign_mir(&mut self, def_id: DefId) -> Option<Rc<Mir<'tcx>>> {
        if let Some(mir) = self.foreign_mirs.get(&def_id).cloned() {
            return Some(mir);
        }
        // Cache the foreign MIR
        let mir = self.tcx.sess.cstore.maybe_get_item_mir(self.tcx, def_id);
        let mir = mir.map(Rc::new);
        if let Some(ref mir) = mir {
            self.foreign_mirs.insert(def_id, mir.clone());
        }

        mir
    }

    fn should_inline(&self, callsite: CallSite<'tcx>,
                     callee_mir: &'a Mir<'tcx>) -> bool {

        let tcx = self.tcx;

        // Don't inline closures
        if callee_mir.upvar_decls.len() > 0 {
            return false;
        }

        // Don't inline calls to trait methods
        // FIXME: Should try to resolve it to a concrete method, and
        // only bail if that isn't possible
        let trait_def = tcx.trait_of_item(callsite.callee);
        if trait_def.is_some() { return false; }

        let attrs = tcx.get_attrs(callsite.callee);
        let hint = attr::find_inline_attr(None, &attrs[..]);

        let hinted = match hint {
            // Just treat inline(always) as a hint for now,
            // there are cases that prevent unwinding that we
            // need to check for first.
            attr::InlineAttr::Always => true,
            attr::InlineAttr::Never => return false,
            attr::InlineAttr::Hint => true,
            attr::InlineAttr::None => false,
        };

        // Only inline local functions if they would be eligible for
        // cross-crate inlining. This ensures that any symbols they
        // use are reachable cross-crate
        // FIXME: This shouldn't be necessary, trans should generate
        // the reachable set from the MIR.
        if callsite.callee.is_local() {
            // No type substs and no inline hint means this function
            // wouldn't be eligible for cross-crate inlining
            if callsite.substs.types().count() == 0 && !hinted {
                return false;
            }

        }

        let mut threshold = if hinted {
            HINT_THRESHOLD
        } else {
            DEFAULT_THRESHOLD
        };

        // Significantly lower the threshold for inlining cold functions
        if attr::contains_name(&attrs[..], "cold") {
            threshold /= 5;
        }

        // Give a bonus functions with a small number of blocks,
        // We normally have two or three blocks for even
        // very small functions.
        if callee_mir.basic_blocks().len() <= 3 {
            threshold += threshold / 4;
        }


        let id = tcx.map.as_local_node_id(callsite.caller).expect("Caller not local");
        let param_env = ty::ParameterEnvironment::for_item(tcx, id);

        let mut first_block = true;
        let mut cost = 0;

        for blk in callee_mir.basic_blocks() {
            for stmt in &blk.statements {
                // Don't count StorageLive/StorageDead in the inlining cost.
                match stmt.kind {
                    StatementKind::StorageLive(_) |
                    StatementKind::StorageDead(_) => {}
                    _ => cost += INSTR_COST
                }
            }
            match blk.terminator().kind {
                TerminatorKind::Drop { ref location, unwind, .. } |
                TerminatorKind::DropAndReplace { ref location, unwind, .. } => {
                    // If the location doesn't actually need dropping, treat it like
                    // a regular goto.
                    let ty = location.ty(&callee_mir, tcx).subst(tcx, callsite.substs);
                    let ty = ty.to_ty(tcx);
                    if tcx.type_needs_drop_given_env(ty, &param_env) {
                        if unwind.is_some() {
                            // FIXME: Should be able to handle this better
                            return false;
                        } else {
                            cost += CALL_PENALTY;
                        }
                    } else {
                        cost += INSTR_COST;
                    }
                }
                // FIXME: Should be able to handle this better
                TerminatorKind::Call   { cleanup: Some(_), .. } |
                TerminatorKind::Assert { cleanup: Some(_), .. } => return false,

                TerminatorKind::Unreachable |
                TerminatorKind::Call { destination: None, .. } if first_block => {
                    // If the function always diverges, don't inline
                    // unless the cost is zero
                    threshold = 0;
                }

                TerminatorKind::Call   { .. } |
                TerminatorKind::Assert { .. } => cost += CALL_PENALTY,
                _ => cost += INSTR_COST
            }
            first_block = false;
        }

        // Count up the cost of local variables and temps, if we know the size
        // use that, otherwise we use a moderately-large dummy cost.

        let ptr_size = tcx.data_layout.pointer_size.bytes();

        for v in &callee_mir.var_decls {
            let ty = v.ty.subst(tcx, callsite.substs);
            // Cost of the var is the size in machine-words, if we know
            // it.
            if let Some(size) = type_size_of(tcx, param_env.clone(), ty) {
                cost += (size / ptr_size) as usize;
            } else {
                cost += UNKNOWN_SIZE_COST;
            }
        }
        for t in &callee_mir.temp_decls {
            let ty = t.ty.subst(tcx, callsite.substs);
            // Cost of the var is the size in machine-words, if we know
            // it.
            if let Some(size) = type_size_of(tcx, param_env.clone(), ty) {
                cost += (size / ptr_size) as usize;
            } else {
                cost += UNKNOWN_SIZE_COST;
            }
        }

        debug!("Inline cost for {:?} is {}", callsite.callee, cost);

        if let attr::InlineAttr::Always = hint {
            true
        } else {
            cost <= threshold
        }
    }


    fn inline_call(&self, callsite: CallSite,
                             caller_mir: &mut Mir<'tcx>, callee_mir: Mir<'tcx>) -> bool {

        // Don't inline a function into itself
        if callsite.caller == callsite.callee { return false; }

        let _task = self.tcx.dep_graph.in_task(DepNode::Mir(callsite.caller));


        let terminator = caller_mir[callsite.bb].terminator.take().unwrap();
        match terminator.kind {
            TerminatorKind::Call {
                func: _, args, destination: Some(destination), cleanup } => {

                debug!("Inlined {:?} into {:?}", callsite.callee, callsite.caller);

                let call_scope = terminator.source_info.scope;
                let call_span = terminator.source_info.span;
                let bb_len = caller_mir.basic_blocks().len();

                let mut var_map = IndexVec::with_capacity(callee_mir.var_decls.len());
                let mut temp_map = IndexVec::with_capacity(callee_mir.temp_decls.len());
                let mut scope_map = IndexVec::with_capacity(callee_mir.visibility_scopes.len());
                let mut promoted_map = IndexVec::with_capacity(callee_mir.promoted.len());

                for mut scope in callee_mir.visibility_scopes {
                    if scope.parent_scope.is_none() {
                        scope.parent_scope = Some(call_scope);
                    }

                    scope.span = call_span;

                    let idx = caller_mir.visibility_scopes.push(scope);
                    scope_map.push(idx);
                }

                for mut var in callee_mir.var_decls {
                    var.source_info.scope = scope_map[var.source_info.scope];
                    let idx = caller_mir.var_decls.push(var);
                    var_map.push(idx);
                }

                for temp in callee_mir.temp_decls {
                    let idx = caller_mir.temp_decls.push(temp);
                    temp_map.push(idx);
                }

                for p in callee_mir.promoted {
                    let idx = caller_mir.promoted.push(p);
                    promoted_map.push(idx);
                }

                // If the call is something like `a[*i] = f(i)`, where
                // `i : &mut usize`, then just duplicating the `a[*i]`
                // Lvalue could result in two different locations if `f`
                // writes to `i`. To prevent this we need to create a temporary
                // borrow of the lvalue and pass the destination as `*temp` instead.
                fn dest_needs_borrow(lval: &Lvalue) -> bool {
                    match *lval {
                        Lvalue::Projection(ref p) => {
                            match p.elem {
                                ProjectionElem::Deref |
                                ProjectionElem::Index(_) => true,
                                _ => dest_needs_borrow(&p.base)
                            }
                        }
                        // Static variables need a borrow because the callee
                        // might modify the same static.
                        Lvalue::Static(_) => true,
                        _ => false
                    }
                }

                let dest = if dest_needs_borrow(&destination.0) {
                    debug!("Creating temp for return destination");
                    let dest = Rvalue::Ref(
                        self.tcx.mk_region(ty::ReErased),
                        BorrowKind::Mut,
                        destination.0);

                    let ty = dest.ty(caller_mir, self.tcx).expect("Rvalue has no type!");

                    let temp = TempDecl { ty: ty };
                    let tmp = caller_mir.temp_decls.push(temp);
                    let tmp = Lvalue::Temp(tmp);

                    let stmt = Statement {
                        source_info: callsite.location,
                        kind: StatementKind::Assign(tmp.clone(), dest)
                    };
                    caller_mir[callsite.bb]
                        .statements.push(stmt);
                    tmp.deref()
                } else {
                    destination.0
                };

                let return_block = destination.1;

                // Copy the arguments if needed.
                let args : Vec<_> = {

                    let tcx = self.tcx;
                    args.iter().map(|a| {
                        if let Operand::Consume(Lvalue::Temp(_)) = *a {
                            // Reuse the operand if it's a temporary already
                            a.clone()
                        } else {
                            debug!("Creating temp for argument");
                            // Otherwise, create a temporary for the arg
                            let arg = Rvalue::Use(a.clone());

                            let ty = arg.ty(caller_mir, tcx).expect("arg has no type!");

                            let temp = TempDecl { ty: ty };
                            let tmp = caller_mir.temp_decls.push(temp);
                            let tmp = Lvalue::Temp(tmp);

                            let stmt = Statement {
                                source_info: callsite.location,
                                kind: StatementKind::Assign(tmp.clone(), arg)
                            };
                            caller_mir[callsite.bb].statements.push(stmt);
                            Operand::Consume(tmp)
                        }
                    }).collect()
                };

                let mut integrator = Integrator {
                    block_idx: bb_len,
                    args: &args,
                    var_map: var_map,
                    tmp_map: temp_map,
                    scope_map: scope_map,
                    promoted_map: promoted_map,
                    inline_location: callsite.location,
                    destination: dest,
                    return_block: return_block,
                    cleanup_block: cleanup
                };


                for (bb, mut block) in callee_mir.basic_blocks.into_iter_enumerated() {
                    integrator.visit_basic_block_data(bb, &mut block);
                    caller_mir.basic_blocks_mut().push(block);
                }

                let terminator = Terminator {
                    source_info: terminator.source_info,
                    kind: TerminatorKind::Goto { target: BasicBlock::new(bb_len) }
                };

                caller_mir[callsite.bb].terminator = Some(terminator);

                true
            }
            kind => {
                caller_mir[callsite.bb].terminator = Some(Terminator {
                    source_info: terminator.source_info,
                    kind: kind
                });
                false
            }
        }
    }
}

fn type_size_of<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, param_env: ty::ParameterEnvironment<'tcx>,
                          ty: Ty<'tcx>) -> Option<u64> {
    tcx.infer_ctxt(None, Some(param_env), traits::Reveal::All).enter(|infcx| {
        ty.layout(&infcx).ok().map(|layout| {
            layout.size(&tcx.data_layout).bytes()
        })
    })
}

struct Integrator<'a, 'tcx: 'a> {
    block_idx: usize,
    args: &'a [Operand<'tcx>],
    var_map: IndexVec<Var, Var>,
    tmp_map: IndexVec<Temp, Temp>,
    scope_map: IndexVec<VisibilityScope, VisibilityScope>,
    promoted_map: IndexVec<Promoted, Promoted>,
    inline_location: SourceInfo,
    destination: Lvalue<'tcx>,
    return_block: BasicBlock,
    cleanup_block: Option<BasicBlock>
}

impl<'a, 'tcx> Integrator<'a, 'tcx> {
    fn update_target(&self, tgt: BasicBlock) -> BasicBlock {
        let new = BasicBlock::new(tgt.index() + self.block_idx);
        debug!("Updating target `{:?}`, new: `{:?}`", tgt, new);
        new
    }
}

impl<'a, 'tcx> MutVisitor<'tcx> for Integrator<'a, 'tcx> {
    fn visit_lvalue(&mut self,
                    lvalue: &mut Lvalue<'tcx>,
                    _ctxt: LvalueContext,
                    _location: Location) {
        match *lvalue {
            Lvalue::Var(ref mut var) => {
                if let Some(v) = self.var_map.get(*var).cloned() {
                    debug!("Replacing {:?} with {:?}", var, v);
                    *var = v;
                }
            }
            Lvalue::Temp(ref mut tmp) => {
                if let Some(t) = self.tmp_map.get(*tmp).cloned() {
                    debug!("Replacing {:?} with {:?}", tmp, t);
                    *tmp = t;
                }
            }
            Lvalue::ReturnPointer => {
                debug!("Replacing return pointer with {:?}", self.destination);
                *lvalue = self.destination.clone();
            }
            Lvalue::Arg(arg) => {
                let idx = arg.index();
                if let Operand::Consume(ref lval) = self.args[idx] {
                    debug!("Replacing {:?} with {:?}", lvalue, lval);
                    *lvalue = lval.clone();
                }
            }
            _ => self.super_lvalue(lvalue, _ctxt, _location)
        }
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, location: Location) {
        if let Operand::Consume(Lvalue::Arg(arg)) = *operand {
            let idx = arg.index();
            let new_arg = self.args[idx].clone();
            debug!("Replacing use of {:?} with {:?}", arg, new_arg);
            *operand = new_arg;
        } else {
            self.super_operand(operand, location);
        }
    }

    fn visit_terminator_kind(&mut self, block: BasicBlock,
                             kind: &mut TerminatorKind<'tcx>, loc: Location) {
        match *kind {
            TerminatorKind::Goto { ref mut target} => {
                *target = self.update_target(*target);
            }
            TerminatorKind::If { ref mut targets, .. } => {
                targets.0 = self.update_target(targets.0);
                targets.1 = self.update_target(targets.1);
            }
            TerminatorKind::Switch { ref mut targets, .. } |
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
                } else {
                    if Some(*target) != self.cleanup_block {
                        *unwind = self.cleanup_block;
                    }
                }

                if Some(*target) == *unwind {
                    *unwind == None;
                }
            }
            TerminatorKind::Call { ref mut destination, ref mut cleanup, .. } => {
                let mut target = None;
                if let Some((_, ref mut tgt)) = *destination {
                    *tgt = self.update_target(*tgt);
                    target = Some(*tgt);
                }
                if let Some(tgt) = *cleanup {
                    *cleanup = Some(self.update_target(tgt));
                } else {
                    *cleanup = self.cleanup_block;
                }

                if target == *cleanup {
                    *cleanup == None;
                }
            }
            TerminatorKind::Assert { ref mut target, ref mut cleanup, .. } => {
                *target = self.update_target(*target);
                if let Some(tgt) = *cleanup {
                    *cleanup = Some(self.update_target(tgt));
                } else {
                    if Some(*target) != self.cleanup_block {
                        *cleanup = self.cleanup_block;
                    }
                }

                if Some(*target) == *cleanup {
                    *cleanup == None;
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
            TerminatorKind::Unreachable => { }
        }

        self.super_terminator_kind(block, kind, loc);
    }
    fn visit_visibility_scope(&mut self, scope: &mut VisibilityScope) {
        *scope = self.scope_map[*scope];
    }
    fn visit_span(&mut self, span: &mut Span) {
        // FIXME: probably shouldn't use the inline location span,
        // but not doing so causes errors
        *span = self.inline_location.span;
    }

    fn visit_literal(&mut self, literal: &mut Literal<'tcx>, loc: Location) {
        if let Literal::Promoted { ref mut index } = *literal {
            if let Some(p) = self.promoted_map.get(*index).cloned() {
                *index = p;
            }
        } else {
            self.super_literal(literal, loc);
        }
    }
}
