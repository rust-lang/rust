// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This pass just dumps MIR at a specified point.

use rustc::ty::TyCtxt;
use rustc::mir::repr::*;
use rustc::mir::transform::{Pass, MirPass, MirSource};
use pretty;

pub struct DumpMir<'a>(pub &'a str);

impl<'b, 'tcx> MirPass<'tcx> for DumpMir<'b> {
    fn run_pass<'a>(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    src: MirSource, mir: &mut Mir<'tcx>) {
        pretty::dump_mir(tcx, self.0, &0, src, mir, None);
    }
}

impl<'b> Pass for DumpMir<'b> {}
