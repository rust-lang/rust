// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::mir::repr as mir;

use common::{self, BlockAndBuilder};

use super::MirContext;
use super::LocalRef;

impl<'bcx, 'tcx> MirContext<'bcx, 'tcx> {
    pub fn trans_statement(&mut self,
                           bcx: BlockAndBuilder<'bcx, 'tcx>,
                           statement: &mir::Statement<'tcx>)
                           -> BlockAndBuilder<'bcx, 'tcx> {
        debug!("trans_statement(statement={:?})", statement);

        let debug_loc = self.debug_loc(statement.source_info);
        debug_loc.apply_to_bcx(&bcx);
        debug_loc.apply(bcx.fcx());
        match statement.kind {
            mir::StatementKind::Assign(ref lvalue, ref rvalue) => {
                if let Some(index) = self.mir.local_index(lvalue) {
                    match self.locals[index] {
                        LocalRef::Lvalue(tr_dest) => {
                            self.trans_rvalue(bcx, tr_dest, rvalue, debug_loc)
                        }
                        LocalRef::Operand(None) => {
                            let (bcx, operand) = self.trans_rvalue_operand(bcx, rvalue,
                                                                           debug_loc);
                            self.locals[index] = LocalRef::Operand(Some(operand));
                            bcx
                        }
                        LocalRef::Operand(Some(_)) => {
                            let ty = self.monomorphized_lvalue_ty(lvalue);

                            if !common::type_is_zero_size(bcx.ccx(), ty) {
                                span_bug!(statement.source_info.span,
                                          "operand {:?} already assigned",
                                          rvalue);
                            } else {
                                // If the type is zero-sized, it's already been set here,
                                // but we still need to make sure we translate the operand
                                self.trans_rvalue_operand(bcx, rvalue, debug_loc).0
                            }
                        }
                    }
                } else {
                    let tr_dest = self.trans_lvalue(&bcx, lvalue);
                    self.trans_rvalue(bcx, tr_dest, rvalue, debug_loc)
                }
            }
        }
    }
}
