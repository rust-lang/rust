// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::bitvec::BitVector;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc::mir::{self, TerminatorKind};
use rustc::mir::traversal;
use rustc::ty::Ty;
use rustc_mir::ssa_analyze;
use rustc::ty::layout::LayoutOf;

use type_of::LayoutLlvmExt;
use super::FunctionCx;

pub fn non_ssa_locals<'a, 'tcx: 'a>(fx: &FunctionCx<'a, 'tcx>) -> BitVector {
    impl<'a, 'tcx: 'a> ssa_analyze::LocalAnalyzerCallbacks<'tcx> for &'a FunctionCx<'a, 'tcx>{
        fn ty_ssa_allowed(&self, ty: Ty<'tcx>) -> bool {
            let layout = self.cx.layout_of(ty);
            if layout.is_llvm_immediate() {
                // These sorts of types are immediates that we can store
                // in an ValueRef without an alloca.
                true
            } else if layout.is_llvm_scalar_pair() {
                // We allow pairs and uses of any of their 2 fields.
                true
            } else {
                // These sorts of types require an alloca. Note that
                // is_llvm_immediate() may *still* be true, particularly
                // for newtypes, but we currently force some types
                // (e.g. structs) into an alloca unconditionally, just so
                // that we don't have to deal with having two pathways
                // (gep vs extractvalue etc).
                false
            }
        }

        fn does_rvalue_create_operand(&self, rvalue: &mir::Rvalue<'tcx>) -> bool {
            self.rvalue_creates_operand(rvalue)
        }
    }

    ssa_analyze::non_ssa_locals(fx.cx.tcx, fx.mir, fx.param_substs, fx)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CleanupKind {
    NotCleanup,
    Funclet,
    Internal { funclet: mir::BasicBlock }
}

impl CleanupKind {
    pub fn funclet_bb(self, for_bb: mir::BasicBlock) -> Option<mir::BasicBlock> {
        match self {
            CleanupKind::NotCleanup => None,
            CleanupKind::Funclet => Some(for_bb),
            CleanupKind::Internal { funclet } => Some(funclet),
        }
    }
}

pub fn cleanup_kinds<'a, 'tcx>(mir: &mir::Mir<'tcx>) -> IndexVec<mir::BasicBlock, CleanupKind> {
    fn discover_masters<'tcx>(result: &mut IndexVec<mir::BasicBlock, CleanupKind>,
                              mir: &mir::Mir<'tcx>) {
        for (bb, data) in mir.basic_blocks().iter_enumerated() {
            match data.terminator().kind {
                TerminatorKind::Goto { .. } |
                TerminatorKind::Resume |
                TerminatorKind::Abort |
                TerminatorKind::Return |
                TerminatorKind::GeneratorDrop |
                TerminatorKind::Unreachable |
                TerminatorKind::SwitchInt { .. } |
                TerminatorKind::Yield { .. } |
                TerminatorKind::FalseEdges { .. } |
                TerminatorKind::FalseUnwind { .. } => {
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

            for &succ in data.terminator().successors() {
                let kind = result[succ];
                debug!("cleanup_kinds: propagating {:?} to {:?}/{:?}",
                       funclet, succ, kind);
                match kind {
                    CleanupKind::NotCleanup => {
                        result[succ] = CleanupKind::Internal { funclet: funclet };
                    }
                    CleanupKind::Funclet => {
                        if funclet != succ {
                            set_successor(funclet, succ);
                        }
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
