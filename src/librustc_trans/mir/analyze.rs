// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An analysis to determine which locals require allocas and
//! which do not.

use rustc_data_structures::bitvec::BitVector;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use rustc::mir::{self, Location, TerminatorKind};
use rustc::mir::visit::{Visitor, LvalueContext};
use rustc::mir::traversal;
use common;
use super::MirContext;
use super::rvalue;

pub fn lvalue_locals<'a, 'tcx>(mircx: &MirContext<'a, 'tcx>) -> BitVector {
    let mir = mircx.mir;
    let mut analyzer = LocalAnalyzer::new(mircx);

    analyzer.visit_mir(mir);

    for (index, ty) in mir.local_decls.iter().map(|l| l.ty).enumerate() {
        let ty = mircx.monomorphize(&ty);
        debug!("local {} has type {:?}", index, ty);
        if ty.is_scalar() ||
            ty.is_unique() ||
            ty.is_region_ptr() ||
            ty.is_simd() ||
            common::type_is_zero_size(mircx.ccx, ty)
        {
            // These sorts of types are immediates that we can store
            // in an ValueRef without an alloca.
            assert!(common::type_is_immediate(mircx.ccx, ty) ||
                    common::type_is_fat_ptr(mircx.ccx, ty));
        } else if common::type_is_imm_pair(mircx.ccx, ty) {
            // We allow pairs and uses of any of their 2 fields.
        } else {
            // These sorts of types require an alloca. Note that
            // type_is_immediate() may *still* be true, particularly
            // for newtypes, but we currently force some types
            // (e.g. structs) into an alloca unconditionally, just so
            // that we don't have to deal with having two pathways
            // (gep vs extractvalue etc).
            analyzer.mark_as_lvalue(mir::Local::new(index));
        }
    }

    analyzer.lvalue_locals
}

struct LocalAnalyzer<'mir, 'a: 'mir, 'tcx: 'a> {
    cx: &'mir MirContext<'a, 'tcx>,
    lvalue_locals: BitVector,
    seen_assigned: BitVector
}

impl<'mir, 'a, 'tcx> LocalAnalyzer<'mir, 'a, 'tcx> {
    fn new(mircx: &'mir MirContext<'a, 'tcx>) -> LocalAnalyzer<'mir, 'a, 'tcx> {
        LocalAnalyzer {
            cx: mircx,
            lvalue_locals: BitVector::new(mircx.mir.local_decls.len()),
            seen_assigned: BitVector::new(mircx.mir.local_decls.len())
        }
    }

    fn mark_as_lvalue(&mut self, local: mir::Local) {
        debug!("marking {:?} as lvalue", local);
        self.lvalue_locals.insert(local.index());
    }

    fn mark_assigned(&mut self, local: mir::Local) {
        if !self.seen_assigned.insert(local.index()) {
            self.mark_as_lvalue(local);
        }
    }
}

impl<'mir, 'a, 'tcx> Visitor<'tcx> for LocalAnalyzer<'mir, 'a, 'tcx> {
    fn visit_assign(&mut self,
                    block: mir::BasicBlock,
                    lvalue: &mir::Lvalue<'tcx>,
                    rvalue: &mir::Rvalue<'tcx>,
                    location: Location) {
        debug!("visit_assign(block={:?}, lvalue={:?}, rvalue={:?})", block, lvalue, rvalue);

        if let mir::Lvalue::Local(index) = *lvalue {
            self.mark_assigned(index);
            if !rvalue::rvalue_creates_operand(rvalue) {
                self.mark_as_lvalue(index);
            }
        } else {
            self.visit_lvalue(lvalue, LvalueContext::Store, location);
        }

        self.visit_rvalue(rvalue, location);
    }

    fn visit_terminator_kind(&mut self,
                             block: mir::BasicBlock,
                             kind: &mir::TerminatorKind<'tcx>,
                             location: Location) {
        match *kind {
            mir::TerminatorKind::Call {
                func: mir::Operand::Constant(mir::Constant {
                    literal: mir::Literal::Item { def_id, .. }, ..
                }),
                ref args, ..
            } if Some(def_id) == self.cx.ccx.tcx().lang_items.box_free_fn() => {
                // box_free(x) shares with `drop x` the property that it
                // is not guaranteed to be statically dominated by the
                // definition of x, so x must always be in an alloca.
                if let mir::Operand::Consume(ref lvalue) = args[0] {
                    self.visit_lvalue(lvalue, LvalueContext::Drop, location);
                }
            }
            _ => {}
        }

        self.super_terminator_kind(block, kind, location);
    }

    fn visit_lvalue(&mut self,
                    lvalue: &mir::Lvalue<'tcx>,
                    context: LvalueContext<'tcx>,
                    location: Location) {
        debug!("visit_lvalue(lvalue={:?}, context={:?})", lvalue, context);

        // Allow uses of projections of immediate pair fields.
        if let mir::Lvalue::Projection(ref proj) = *lvalue {
            if let mir::Lvalue::Local(_) = proj.base {
                let ty = proj.base.ty(self.cx.mir, self.cx.ccx.tcx());

                let ty = self.cx.monomorphize(&ty.to_ty(self.cx.ccx.tcx()));
                if common::type_is_imm_pair(self.cx.ccx, ty) {
                    if let mir::ProjectionElem::Field(..) = proj.elem {
                        if let LvalueContext::Consume = context {
                            return;
                        }
                    }
                }
            }
        }

        if let mir::Lvalue::Local(index) = *lvalue {
            match context {
                LvalueContext::Call => {
                    self.mark_assigned(index);
                }

                LvalueContext::StorageLive |
                LvalueContext::StorageDead |
                LvalueContext::Consume => {}

                LvalueContext::Store |
                LvalueContext::Inspect |
                LvalueContext::Borrow { .. } |
                LvalueContext::Projection(..) => {
                    self.mark_as_lvalue(index);
                }

                LvalueContext::Drop => {
                    let ty = lvalue.ty(self.cx.mir, self.cx.ccx.tcx());
                    let ty = self.cx.monomorphize(&ty.to_ty(self.cx.ccx.tcx()));

                    // Only need the lvalue if we're actually dropping it.
                    if self.cx.ccx.shared().type_needs_drop(ty) {
                        self.mark_as_lvalue(index);
                    }
                }
            }
        }

        // A deref projection only reads the pointer, never needs the lvalue.
        if let mir::Lvalue::Projection(ref proj) = *lvalue {
            if let mir::ProjectionElem::Deref = proj.elem {
                return self.visit_lvalue(&proj.base, LvalueContext::Consume, location);
            }
        }

        self.super_lvalue(lvalue, context, location);
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CleanupKind {
    NotCleanup,
    Funclet,
    Internal { funclet: mir::BasicBlock }
}

pub fn cleanup_kinds<'a, 'tcx>(mir: &mir::Mir<'tcx>) -> IndexVec<mir::BasicBlock, CleanupKind> {
    fn discover_masters<'tcx>(result: &mut IndexVec<mir::BasicBlock, CleanupKind>,
                              mir: &mir::Mir<'tcx>) {
        for (bb, data) in mir.basic_blocks().iter_enumerated() {
            match data.terminator().kind {
                TerminatorKind::Goto { .. } |
                TerminatorKind::Resume |
                TerminatorKind::Return |
                TerminatorKind::Unreachable |
                TerminatorKind::If { .. } |
                TerminatorKind::Switch { .. } |
                TerminatorKind::SwitchInt { .. } => {
                    /* nothing to do */
                }
                TerminatorKind::Call { cleanup: unwind, .. } |
                TerminatorKind::Assert { cleanup: unwind, .. } |
                TerminatorKind::DropAndReplace { unwind, .. } |
                TerminatorKind::Drop { unwind, .. } => {
                    if let Some(unwind) = unwind {
                        debug!("cleanup_kinds: {:?}/{:?} registering {:?} as funclet",
                               bb, data, unwind);
                        result[unwind] = CleanupKind::Funclet;
                    }
                }
            }
        }
    }

    fn propagate<'tcx>(result: &mut IndexVec<mir::BasicBlock, CleanupKind>,
                       mir: &mir::Mir<'tcx>) {
        let mut funclet_succs = IndexVec::from_elem(None, mir.basic_blocks());

        let mut set_successor = |funclet: mir::BasicBlock, succ| {
            match funclet_succs[funclet] {
                ref mut s @ None => {
                    debug!("set_successor: updating successor of {:?} to {:?}",
                           funclet, succ);
                    *s = Some(succ);
                },
                Some(s) => if s != succ {
                    span_bug!(mir.span, "funclet {:?} has 2 parents - {:?} and {:?}",
                              funclet, s, succ);
                }
            }
        };

        for (bb, data) in traversal::reverse_postorder(mir) {
            let funclet = match result[bb] {
                CleanupKind::NotCleanup => continue,
                CleanupKind::Funclet => bb,
                CleanupKind::Internal { funclet } => funclet,
            };

            debug!("cleanup_kinds: {:?}/{:?}/{:?} propagating funclet {:?}",
                   bb, data, result[bb], funclet);

            for &succ in data.terminator().successors().iter() {
                let kind = result[succ];
                debug!("cleanup_kinds: propagating {:?} to {:?}/{:?}",
                       funclet, succ, kind);
                match kind {
                    CleanupKind::NotCleanup => {
                        result[succ] = CleanupKind::Internal { funclet: funclet };
                    }
                    CleanupKind::Funclet => {
                        set_successor(funclet, succ);
                    }
                    CleanupKind::Internal { funclet: succ_funclet } => {
                        if funclet != succ_funclet {
                            // `succ` has 2 different funclet going into it, so it must
                            // be a funclet by itself.

                            debug!("promoting {:?} to a funclet and updating {:?}", succ,
                                   succ_funclet);
                            result[succ] = CleanupKind::Funclet;
                            set_successor(succ_funclet, succ);
                            set_successor(funclet, succ);
                        }
                    }
                }
            }
        }
    }

    let mut result = IndexVec::from_elem(CleanupKind::NotCleanup, mir.basic_blocks());

    discover_masters(&mut result, mir);
    propagate(&mut result, mir);
    debug!("cleanup_kinds: result={:?}", result);
    result
}
