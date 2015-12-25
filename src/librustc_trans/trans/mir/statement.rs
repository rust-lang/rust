// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::middle::ty::LvaluePreference;
use rustc::mir::repr as mir;
use trans::common::Block;
use trans::debuginfo::DebugLoc;
use trans::glue;

use super::MirContext;
use super::TempRef;

impl<'bcx, 'tcx> MirContext<'bcx, 'tcx> {
    pub fn trans_statement(&mut self,
                           bcx: Block<'bcx, 'tcx>,
                           statement: &mir::Statement<'tcx>)
                           -> Block<'bcx, 'tcx> {
        debug!("trans_statement(statement={:?})", statement);

        match statement.kind {
            mir::StatementKind::Assign(ref lvalue, ref rvalue) => {
                match *lvalue {
                    mir::Lvalue::Temp(index) => {
                        let index = index as usize;
                        match self.temps[index as usize] {
                            TempRef::Lvalue(tr_dest) => {
                                self.trans_rvalue(bcx, tr_dest, rvalue)
                            }
                            TempRef::Operand(None) => {
                                let (bcx, operand) = self.trans_rvalue_operand(bcx, rvalue);
                                self.temps[index] = TempRef::Operand(Some(operand));
                                bcx
                            }
                            TempRef::Operand(Some(_)) => {
                                bcx.tcx().sess.span_bug(
                                    statement.span,
                                    &format!("operand {:?} already assigned", rvalue));
                            }
                        }
                    }
                    _ => {
                        let tr_dest = self.trans_lvalue(bcx, lvalue);
                        self.trans_rvalue(bcx, tr_dest, rvalue)
                    }
                }
            }

            mir::StatementKind::Drop(mir::DropKind::Deep, ref lvalue) => {
                let tr_lvalue = self.trans_lvalue(bcx, lvalue);
                let ty = tr_lvalue.ty.to_ty(bcx.tcx());
                glue::drop_ty(bcx, tr_lvalue.llval, ty, DebugLoc::None)
            }

            mir::StatementKind::Drop(mir::DropKind::Free, ref lvalue) => {
                let tr_lvalue = self.trans_lvalue(bcx, lvalue);
                let ty = tr_lvalue.ty.to_ty(bcx.tcx());
                let content_ty = ty.builtin_deref(true, LvaluePreference::NoPreference);
                let content_ty = content_ty.unwrap().ty;
                glue::trans_exchange_free_ty(bcx, tr_lvalue.llval, content_ty, DebugLoc::None)
            }
        }
    }
}
