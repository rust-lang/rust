// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use util::nodemap::NodeMap;
use mir::repr::Mir;
use mir::transform::MirPass;
use middle::ty;

pub struct MirMap<'tcx> {
    pub map: NodeMap<Mir<'tcx>>,
}

impl<'tcx> MirMap<'tcx> {
    pub fn run_passes(&mut self, passes: &mut [Box<MirPass>], tcx: &ty::ctxt<'tcx>) {
        for (_, ref mut mir) in &mut self.map {
            for pass in &mut *passes {
                pass.run_on_mir(mir, tcx)
            }
        }
    }
}
