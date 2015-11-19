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

use rustc_data_structures::fnv::FnvHashSet;
use rustc::mir::repr as mir;
use rustc_mir::visit::{Visitor, LvalueContext};
use trans::common::{self, Block};
use super::rvalue;

pub fn lvalue_temps<'bcx,'tcx>(bcx: Block<'bcx,'tcx>,
                               mir: &mir::Mir<'tcx>)
                               -> FnvHashSet<usize> {
    let mut analyzer = TempAnalyzer::new();

    analyzer.visit_mir(mir);

    for (index, temp_decl) in mir.temp_decls.iter().enumerate() {
        let ty = bcx.monomorphize(&temp_decl.ty);
        debug!("temp {:?} has type {:?}", index, ty);
        if ty.is_scalar() ||
            ty.is_unique() ||
            ty.is_region_ptr() ||
            ty.is_simd()
        {
            // These sorts of types are immediates that we can store
            // in an ValueRef without an alloca.
            assert!(common::type_is_immediate(bcx.ccx(), ty) ||
                    common::type_is_fat_ptr(bcx.tcx(), ty));
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

struct TempAnalyzer {
    lvalue_temps: FnvHashSet<usize>,
}

impl TempAnalyzer {
    fn new() -> TempAnalyzer {
        TempAnalyzer { lvalue_temps: FnvHashSet() }
    }

    fn mark_as_lvalue(&mut self, temp: usize) {
        debug!("marking temp {} as lvalue", temp);
        self.lvalue_temps.insert(temp);
    }
}

impl<'tcx> Visitor<'tcx> for TempAnalyzer {
    fn visit_assign(&mut self,
                    block: mir::BasicBlock,
                    lvalue: &mir::Lvalue<'tcx>,
                    rvalue: &mir::Rvalue<'tcx>) {
        debug!("visit_assign(block={:?}, lvalue={:?}, rvalue={:?})", block, lvalue, rvalue);

        match *lvalue {
            mir::Lvalue::Temp(index) => {
                if !rvalue::rvalue_creates_operand(rvalue) {
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

        match *lvalue {
            mir::Lvalue::Temp(index) => {
                match context {
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
