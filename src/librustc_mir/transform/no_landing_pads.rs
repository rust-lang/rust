// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This pass removes the unwind branch of all the terminators when the no-landing-pads option is
//! specified.

use rustc::ty::TyCtxt;
use rustc::mir::repr::*;
use rustc::mir::visit::MutVisitor;
use rustc::mir::transform::{Pass, MirPass};
use syntax::ast::NodeId;

pub struct NoLandingPads;

impl<'tcx> MutVisitor<'tcx> for NoLandingPads {
    fn visit_terminator(&mut self, bb: BasicBlock, terminator: &mut Terminator<'tcx>) {
        match terminator.kind {
            TerminatorKind::Goto { .. } |
            TerminatorKind::Resume |
            TerminatorKind::Return |
            TerminatorKind::If { .. } |
            TerminatorKind::Switch { .. } |
            TerminatorKind::SwitchInt { .. } => {
                /* nothing to do */
            },
            TerminatorKind::Drop { ref mut unwind, .. } => {
                unwind.take();
            },
            TerminatorKind::Call { ref mut cleanup, .. } => {
                cleanup.take();
            },
        }
        self.super_terminator(bb, terminator);
    }
}

impl<'tcx> MirPass<'tcx> for NoLandingPads {
    fn run_pass(&mut self, tcx: &TyCtxt<'tcx>, _: NodeId, mir: &mut Mir<'tcx>) {
        if tcx.sess.no_landing_pads() {
            self.visit_mir(mir);
        }
    }
}

impl Pass for NoLandingPads {}
