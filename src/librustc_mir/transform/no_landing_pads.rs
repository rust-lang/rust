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
use rustc::mir::*;
use rustc::mir::visit::MutVisitor;
use rustc::mir::transform::{Pass, MirPass, MirSource};

pub struct NoLandingPads;

impl<'tcx> MutVisitor<'tcx> for NoLandingPads {
    fn visit_terminator(&mut self,
                        bb: Block,
                        terminator: &mut Terminator<'tcx>,
                        location: Location) {
        match terminator.kind {
            TerminatorKind::Goto { .. } |
            TerminatorKind::Resume |
            TerminatorKind::Return |
            TerminatorKind::Unreachable |
            TerminatorKind::SwitchInt { .. } => {
                /* nothing to do */
            },
            TerminatorKind::DropAndReplace { ref mut unwind, .. } |
            TerminatorKind::Drop { ref mut unwind, .. } => {
                unwind.take();
            },
        }
        self.super_terminator(bb, terminator, location);
    }

    fn visit_statement(
        &mut self,
        bb: Block,
        statement: &mut Statement<'tcx>,
        location: Location) {
        match statement.kind {
            StatementKind::Assign(..) |
            StatementKind::SetDiscriminant { .. } |
            StatementKind::StorageLive(..) |
            StatementKind::StorageDead(..) |
            StatementKind::InlineAsm { .. } |
            StatementKind::Nop => {
                /* nothing to do */
            },
            StatementKind::Call { ref mut cleanup, .. } |
            StatementKind::Assert { ref mut cleanup, .. } => {
                cleanup.take();
            }
        }
        self.super_statement(bb, statement, location);
    }
}

pub fn no_landing_pads<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, mir: &mut Mir<'tcx>) {
    if tcx.sess.no_landing_pads() {
        NoLandingPads.visit_mir(mir);
    }
}

impl<'tcx> MirPass<'tcx> for NoLandingPads {
    fn run_pass<'a>(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    _: MirSource, mir: &mut Mir<'tcx>) {
        no_landing_pads(tcx, mir)
    }
}

impl Pass for NoLandingPads {}
