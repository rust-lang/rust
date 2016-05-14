// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Routines for manipulating the control-flow graph.

use build::Location;
use rustc::mir::repr::*;

pub trait CfgExt<'tcx> {
    fn current_location(&mut self, block: BasicBlock) -> Location;

}

impl<'tcx> CfgExt<'tcx> for CFG<'tcx> {
    fn current_location(&mut self, block: BasicBlock) -> Location {
        let index = self[block].statements.len();
        Location { block: block, statement_index: index }
    }
}
