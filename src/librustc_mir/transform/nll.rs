// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty::TyCtxt;
use rustc::mir::Mir;
use rustc::mir::visit::MutVisitor;
use rustc::mir::transform::{MirPass, MirSource};

#[allow(dead_code)]
struct NLLVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
}

impl<'a, 'tcx> NLLVisitor<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Self {
        NLLVisitor {
            tcx: tcx
        }
    }
}

impl<'a, 'tcx> MutVisitor<'tcx> for NLLVisitor<'a, 'tcx> {
    // FIXME: Nashenas88: implement me!
}

// MIR Pass for non-lexical lifetimes
pub struct NLL;

impl MirPass for NLL {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          _: MirSource,
                          mir: &mut Mir<'tcx>) {
        if tcx.sess.opts.debugging_opts.nll {
            // Clone mir so we can mutate it without disturbing the rest
            // of the compiler
            NLLVisitor::new(tcx).visit_mir(&mut mir.clone());
        }
    }
}