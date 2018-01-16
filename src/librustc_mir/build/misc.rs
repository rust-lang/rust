// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Miscellaneous builder routines that are not specific to building any particular
//! kind of thing.

use build::Builder;

use rustc_const_math::{ConstInt, ConstUsize, ConstIsize};
use rustc::middle::const_val::ConstVal;
use rustc::ty::{self, Ty};
use rustc::mir::interpret::{Value, PrimVal};

use rustc::mir::*;
use syntax::ast;
use syntax_pos::{Span, DUMMY_SP};

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {
    /// Add a new temporary value of type `ty` storing the result of
    /// evaluating `expr`.
    ///
    /// NB: **No cleanup is scheduled for this temporary.** You should
    /// call `schedule_drop` once the temporary is initialized.
    pub fn temp(&mut self, ty: Ty<'tcx>, span: Span) -> Place<'tcx> {
        let temp = self.local_decls.push(LocalDecl::new_temp(ty, span));
        let place = Place::Local(temp);
        debug!("temp: created temp {:?} with type {:?}",
               place, self.local_decls[temp].ty);
        place
    }

    pub fn literal_operand(&mut self,
                           span: Span,
                           ty: Ty<'tcx>,
                           literal: Literal<'tcx>)
                           -> Operand<'tcx> {
        let constant = box Constant {
            span,
            ty,
            literal,
        };
        Operand::Constant(constant)
    }

    pub fn unit_rvalue(&mut self) -> Rvalue<'tcx> {
        Rvalue::Aggregate(box AggregateKind::Tuple, vec![])
    }

    // Returns a zero literal operand for the appropriate type, works for
    // bool, char and integers.
    pub fn zero_literal(&mut self, span: Span, ty: Ty<'tcx>) -> Operand<'tcx> {
        let literal = match ty.sty {
            ty::TyBool => {
                self.hir.false_literal()
            }
            ty::TyChar => {
                Literal::Value {
                    value: self.hir.tcx().mk_const(ty::Const {
                        val: if self.hir.tcx().sess.opts.debugging_opts.miri {
                            ConstVal::Value(Value::ByVal(PrimVal::Bytes(0)))
                        } else {
                            ConstVal::Char('\0')
                        },
                        ty
                    })
                }
            }
            ty::TyUint(ity) => {
                let val = match ity {
                    ast::UintTy::U8  => ConstInt::U8(0),
                    ast::UintTy::U16 => ConstInt::U16(0),
                    ast::UintTy::U32 => ConstInt::U32(0),
                    ast::UintTy::U64 => ConstInt::U64(0),
                    ast::UintTy::U128 => ConstInt::U128(0),
                    ast::UintTy::Usize => {
                        let uint_ty = self.hir.tcx().sess.target.usize_ty;
                        let val = ConstUsize::new(0, uint_ty).unwrap();
                        ConstInt::Usize(val)
                    }
                };

                Literal::Value {
                    value: self.hir.tcx().mk_const(ty::Const {
                        val: if self.hir.tcx().sess.opts.debugging_opts.miri {
                            ConstVal::Value(Value::ByVal(PrimVal::Bytes(0)))
                        } else {
                            ConstVal::Integral(val)
                        },
                        ty
                    })
                }
            }
            ty::TyInt(ity) => {
                let val = match ity {
                    ast::IntTy::I8  => ConstInt::I8(0),
                    ast::IntTy::I16 => ConstInt::I16(0),
                    ast::IntTy::I32 => ConstInt::I32(0),
                    ast::IntTy::I64 => ConstInt::I64(0),
                    ast::IntTy::I128 => ConstInt::I128(0),
                    ast::IntTy::Isize => {
                        let int_ty = self.hir.tcx().sess.target.isize_ty;
                        let val = ConstIsize::new(0, int_ty).unwrap();
                        ConstInt::Isize(val)
                    }
                };

                Literal::Value {
                    value: self.hir.tcx().mk_const(ty::Const {
                        val: if self.hir.tcx().sess.opts.debugging_opts.miri {
                            ConstVal::Value(Value::ByVal(PrimVal::Bytes(0)))
                        } else {
                            ConstVal::Integral(val)
                        },
                        ty
                    })
                }
            }
            _ => {
                span_bug!(span, "Invalid type for zero_literal: `{:?}`", ty)
            }
        };

        self.literal_operand(span, ty, literal)
    }

    pub fn push_usize(&mut self,
                      block: BasicBlock,
                      source_info: SourceInfo,
                      value: u64)
                      -> Place<'tcx> {
        let usize_ty = self.hir.usize_ty();
        let temp = self.temp(usize_ty, source_info.span);
        self.cfg.push_assign_constant(
            block, source_info, &temp,
            Constant {
                span: source_info.span,
                ty: self.hir.usize_ty(),
                literal: self.hir.usize_literal(value),
            });
        temp
    }

    pub fn consume_by_copy_or_move(&self, place: Place<'tcx>) -> Operand<'tcx> {
        let tcx = self.hir.tcx();
        let ty = place.ty(&self.local_decls, tcx).to_ty(tcx);
        if self.hir.type_moves_by_default(ty, DUMMY_SP) {
            Operand::Move(place)
        } else {
            Operand::Copy(place)
        }
    }
}
