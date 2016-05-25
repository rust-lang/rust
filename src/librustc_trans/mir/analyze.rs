// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An analysis to determine which temporaries require allocas and
//! which do not.

use rustc_data_structures::bitvec::BitVector;
use rustc::mir::repr as mir;
use rustc::mir::repr::TerminatorKind;
use rustc::mir::visit::{Visitor, LvalueContext};
use rustc_mir::traversal;
use common::{self, Block, BlockAndBuilder};
use super::rvalue;

pub fn lvalue_temps<'bcx,'tcx>(bcx: Block<'bcx,'tcx>,
                               mir: &mir::Mir<'tcx>) -> BitVector {
    let bcx = bcx.build();
    let mut analyzer = TempAnalyzer::new(mir, &bcx, mir.temp_decls.len());

    analyzer.visit_mir(mir);

    for (index, temp_decl) in mir.temp_decls.iter().enumerate() {
        let ty = bcx.monomorphize(&temp_decl.ty);
        debug!("temp {:?} has type {:?}", index, ty);
        if ty.is_scalar() ||
            ty.is_unique() ||
            ty.is_region_ptr() ||
            ty.is_simd() ||
            common::type_is_zero_size(bcx.ccx(), ty)
        {
            // These sorts of types are immediates that we can store
            // in an ValueRef without an alloca.
            assert!(common::type_is_immediate(bcx.ccx(), ty) ||
                    common::type_is_fat_ptr(bcx.tcx(), ty));
        } else if common::type_is_imm_pair(bcx.ccx(), ty) {
            // We allow pairs and uses of any of their 2 fields.
        } else {
            // These sorts of types require an alloca. Note that
            // type_is_immediate() may *still* be true, particularly
            // for newtypes, but we currently force some types
            // (e.g. structs) into an alloca unconditionally, just so
            // that we don't have to deal with having two pathways
            // (gep vs extractvalue etc).
            analyzer.mark_as_lvalue(index);
        }
    }

    analyzer.lvalue_temps
}

struct TempAnalyzer<'mir, 'bcx: 'mir, 'tcx: 'bcx> {
    mir: &'mir mir::Mir<'tcx>,
    bcx: &'mir BlockAndBuilder<'bcx, 'tcx>,
    lvalue_temps: BitVector,
    seen_assigned: BitVector
}

impl<'mir, 'bcx, 'tcx> TempAnalyzer<'mir, 'bcx, 'tcx> {
    fn new(mir: &'mir mir::Mir<'tcx>,
           bcx: &'mir BlockAndBuilder<'bcx, 'tcx>,
           temp_count: usize) -> TempAnalyzer<'mir, 'bcx, 'tcx> {
        TempAnalyzer {
            mir: mir,
            bcx: bcx,
            lvalue_temps: BitVector::new(temp_count),
            seen_assigned: BitVector::new(temp_count)
        }
    }

    fn mark_as_lvalue(&mut self, temp: usize) {
        debug!("marking temp {} as lvalue", temp);
        self.lvalue_temps.insert(temp);
    }

    fn mark_assigned(&mut self, temp: usize) {
        if !self.seen_assigned.insert(temp) {
            self.mark_as_lvalue(temp);
        }
    }
}

impl<'mir, 'bcx, 'tcx> Visitor<'tcx> for TempAnalyzer<'mir, 'bcx, 'tcx> {
    fn visit_assign(&mut self,
                    block: mir::BasicBlock,
                    lvalue: &mir::Lvalue<'tcx>,
                    rvalue: &mir::Rvalue<'tcx>) {
        debug!("visit_assign(block={:?}, lvalue={:?}, rvalue={:?})", block, lvalue, rvalue);

        match *lvalue {
            mir::Lvalue::Temp(index) => {
                self.mark_assigned(index as usize);
                if !rvalue::rvalue_creates_operand(self.mir, self.bcx, rvalue) {
                    self.mark_as_lvalue(index as usize);
                }
            }
            _ => {
                self.visit_lvalue(lvalue, LvalueContext::Store);
            }
        }

        self.visit_rvalue(rvalue);
    }

    fn visit_lvalue(&mut self,
                    lvalue: &mir::Lvalue<'tcx>,
                    context: LvalueContext) {
        debug!("visit_lvalue(lvalue={:?}, context={:?})", lvalue, context);

        // Allow uses of projections of immediate pair fields.
        if let mir::Lvalue::Projection(ref proj) = *lvalue {
            if let mir::Lvalue::Temp(index) = proj.base {
                let ty = self.mir.temp_decls[index as usize].ty;
                let ty = self.bcx.monomorphize(&ty);
                if common::type_is_imm_pair(self.bcx.ccx(), ty) {
                    if let mir::ProjectionElem::Field(..) = proj.elem {
                        if let LvalueContext::Consume = context {
                            return;
                        }
                    }
                }
            }
        }

        match *lvalue {
            mir::Lvalue::Temp(index) => {
                match context {
                    LvalueContext::Call => {
                        self.mark_assigned(index as usize);
                    }
                    LvalueContext::Consume => {
                    }
                    LvalueContext::Store |
                    LvalueContext::Drop |
                    LvalueContext::Inspect |
                    LvalueContext::Borrow { .. } |
                    LvalueContext::Slice { .. } |
                    LvalueContext::Projection => {
                        self.mark_as_lvalue(index as usize);
                    }
                }
            }
            _ => {
            }
        }

        self.super_lvalue(lvalue, context);
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CleanupKind {
    NotCleanup,
    Funclet,
    Internal { funclet: mir::BasicBlock }
}

pub fn cleanup_kinds<'bcx,'tcx>(_bcx: Block<'bcx,'tcx>,
                                mir: &mir::Mir<'tcx>)
                                -> Vec<CleanupKind>
{
    fn discover_masters<'tcx>(result: &mut [CleanupKind], mir: &mir::Mir<'tcx>) {
        for bb in mir.all_basic_blocks() {
            let data = mir.basic_block_data(bb);
            match data.terminator().kind {
                TerminatorKind::Goto { .. } |
                TerminatorKind::Resume |
                TerminatorKind::Return |
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
                        result[unwind.index()] = CleanupKind::Funclet;
                    }
                }
            }
        }
    }

    fn propagate<'tcx>(result: &mut [CleanupKind], mir: &mir::Mir<'tcx>) {
        let mut funclet_succs : Vec<_> =
            mir.all_basic_blocks().iter().map(|_| None).collect();

        let mut set_successor = |funclet: mir::BasicBlock, succ| {
            match funclet_succs[funclet.index()] {
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
            let funclet = match result[bb.index()] {
                CleanupKind::NotCleanup => continue,
                CleanupKind::Funclet => bb,
                CleanupKind::Internal { funclet } => funclet,
            };

            debug!("cleanup_kinds: {:?}/{:?}/{:?} propagating funclet {:?}",
                   bb, data, result[bb.index()], funclet);

            for &succ in data.terminator().successors().iter() {
                let kind = result[succ.index()];
                debug!("cleanup_kinds: propagating {:?} to {:?}/{:?}",
                       funclet, succ, kind);
                match kind {
                    CleanupKind::NotCleanup => {
                        result[succ.index()] = CleanupKind::Internal { funclet: funclet };
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
                            result[succ.index()] = CleanupKind::Funclet;
                            set_successor(succ_funclet, succ);
                            set_successor(funclet, succ);
                        }
                    }
                }
            }
        }
    }

    let mut result : Vec<_> =
        mir.all_basic_blocks().iter().map(|_| CleanupKind::NotCleanup).collect();

    discover_masters(&mut result, mir);
    propagate(&mut result, mir);
    debug!("cleanup_kinds: result={:?}", result);
    result
}
