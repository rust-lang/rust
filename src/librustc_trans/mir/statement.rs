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

use asm;
use builder::Builder;

use super::FunctionCx;
use super::LocalRef;

impl<'a, 'tcx> FunctionCx<'a, 'tcx> {
    pub fn trans_statement(&mut self,
                           bx: Builder<'a, 'tcx>,
                           statement: &mir::Statement<'tcx>)
                           -> Builder<'a, 'tcx> {
        debug!("trans_statement(statement={:?})", statement);

        self.set_debug_loc(&bx, statement.source_info);
        match statement.kind {
            mir::StatementKind::Assign(ref place, ref rvalue) => {
                if let mir::Place::Local(index) = *place {
                    match self.locals[index] {
                        LocalRef::Place(tr_dest) => {
                            self.trans_rvalue(bx, tr_dest, rvalue)
                        }
                        LocalRef::Operand(None) => {
                            let (bx, operand) = self.trans_rvalue_operand(bx, rvalue);
                            self.locals[index] = LocalRef::Operand(Some(operand));
                            bx
                        }
                        LocalRef::Operand(Some(op)) => {
                            if !op.layout.is_zst() {
                                span_bug!(statement.source_info.span,
                                          "operand {:?} already assigned",
                                          rvalue);
                            }

                            // If the type is zero-sized, it's already been set here,
                            // but we still need to make sure we translate the operand
                            self.trans_rvalue_operand(bx, rvalue).0
                        }
                    }
                } else {
                    let tr_dest = self.trans_place(&bx, place);
                    self.trans_rvalue(bx, tr_dest, rvalue)
                }
            }
            mir::StatementKind::SetDiscriminant{ref place, variant_index} => {
                self.trans_place(&bx, place)
                    .trans_set_discr(&bx, variant_index);
                bx
            }
            mir::StatementKind::StorageLive(local) => {
                if let LocalRef::Place(tr_place) = self.locals[local] {
                    tr_place.storage_live(&bx);
                }
                bx
            }
            mir::StatementKind::StorageDead(local) => {
                if let LocalRef::Place(tr_place) = self.locals[local] {
                    tr_place.storage_dead(&bx);
                }
                bx
            }
            mir::StatementKind::InlineAsm { ref asm, ref outputs, ref inputs } => {
                let outputs = outputs.iter().map(|output| {
                    self.trans_place(&bx, output)
                }).collect();

                let input_vals = inputs.iter().map(|input| {
                    self.trans_operand(&bx, input).immediate()
                }).collect();

                asm::trans_inline_asm(&bx, asm, outputs, input_vals);
                bx
            }
            mir::StatementKind::EndRegion(_) |
            mir::StatementKind::Validate(..) |
            mir::StatementKind::Nop => bx,
        }
    }
}
