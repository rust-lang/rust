// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::mir;

use base;
use common;
use builder::Builder;

use super::MirContext;
use super::LocalRef;
use super::super::adt;
use super::super::disr::Disr;

impl<'a, 'tcx> MirContext<'a, 'tcx> {
    pub fn trans_statement(&mut self,
                           bcx: Builder<'a, 'tcx>,
                           statement: &mir::Statement<'tcx>)
                           -> Builder<'a, 'tcx> {
        debug!("trans_statement(statement={:?})", statement);

        self.set_debug_loc(&bcx, statement.source_info);
        match statement.kind {
            mir::StatementKind::Assign(ref lvalue, ref rvalue) => {
                if let mir::Lvalue::Local(index) = *lvalue {
                    match self.locals[index] {
                        LocalRef::Lvalue(tr_dest) => {
                            self.trans_rvalue(bcx, tr_dest, rvalue)
                        }
                        LocalRef::Operand(None) => {
                            let (bcx, operand) = self.trans_rvalue_operand(bcx, rvalue);
                            self.locals[index] = LocalRef::Operand(Some(operand));
                            bcx
                        }
                        LocalRef::Operand(Some(_)) => {
                            let ty = self.monomorphized_lvalue_ty(lvalue);

                            if !common::type_is_zero_size(bcx.ccx, ty) {
                                span_bug!(statement.source_info.span,
                                          "operand {:?} already assigned",
                                          rvalue);
                            } else {
                                // If the type is zero-sized, it's already been set here,
                                // but we still need to make sure we translate the operand
                                self.trans_rvalue_operand(bcx, rvalue).0
                            }
                        }
                    }
                } else {
                    let tr_dest = self.trans_lvalue(&bcx, lvalue);
                    self.trans_rvalue(bcx, tr_dest, rvalue)
                }
            }
            mir::StatementKind::SetDiscriminant{ref lvalue, variant_index} => {
                let ty = self.monomorphized_lvalue_ty(lvalue);
                let lvalue_transed = self.trans_lvalue(&bcx, lvalue);
                adt::trans_set_discr(&bcx,
                    ty,
                    lvalue_transed.llval,
                    Disr::from(variant_index));
                bcx
            }
            mir::StatementKind::StorageLive(ref lvalue) => {
                self.trans_storage_liveness(bcx, lvalue, base::Lifetime::Start)
            }
            mir::StatementKind::StorageDead(ref lvalue) => {
                self.trans_storage_liveness(bcx, lvalue, base::Lifetime::End)
            }
            mir::StatementKind::Nop => bcx,
        }
    }

    fn trans_storage_liveness(&self,
                              bcx: Builder<'a, 'tcx>,
                              lvalue: &mir::Lvalue<'tcx>,
                              intrinsic: base::Lifetime)
                              -> Builder<'a, 'tcx> {
        if let mir::Lvalue::Local(index) = *lvalue {
            if let LocalRef::Lvalue(tr_lval) = self.locals[index] {
                intrinsic.call(&bcx, tr_lval.llval);
            }
        }
        bcx
    }
}
