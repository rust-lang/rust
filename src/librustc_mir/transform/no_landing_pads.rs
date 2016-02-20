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

use rustc::middle::ty;
use rustc::mir::repr::*;
use rustc::mir::visit::MutVisitor;
use rustc::mir::transform::MirPass;

pub struct NoLandingPads;

impl<'tcx> MutVisitor<'tcx> for NoLandingPads {
    fn visit_terminator(&mut self, bb: BasicBlock, terminator: &mut Terminator<'tcx>) {
        match *terminator {
            Terminator::Goto { .. } |
            Terminator::Resume |
            Terminator::Return |
            Terminator::If { .. } |
            Terminator::Switch { .. } |
            Terminator::SwitchInt { .. } => {
                /* nothing to do */
            },
            Terminator::Drop { ref mut unwind, .. } => {
                unwind.take();
            },
            Terminator::Call { ref mut cleanup, .. } => {
                cleanup.take();
            },
        }
        self.super_terminator(bb, terminator);
    }
}

impl MirPass for NoLandingPads {
    fn run_on_mir<'tcx>(&mut self, mir: &mut Mir<'tcx>, tcx: &ty::ctxt<'tcx>) {
        if tcx.sess.no_landing_pads() {
            self.visit_mir(mir);
        }
    }
}
