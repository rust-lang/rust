// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Methods for the various MIR types. These are intended for use after
 * building is complete.
 */

use repr::*;
use rustc::middle::subst::Substs;
use rustc::middle::ty::{self, AdtDef, Ty};
use rustc_front::hir;

#[derive(Copy, Clone, Debug)]
pub enum LvalueTy<'tcx> {
    /// Normal type.
    Ty { ty: Ty<'tcx> },

    /// Downcast to a particular variant of an enum.
    Downcast { adt_def: AdtDef<'tcx>,
               substs: &'tcx Substs<'tcx>,
               variant_index: usize },
}

impl<'tcx> LvalueTy<'tcx> {
    pub fn from_ty(ty: Ty<'tcx>) -> LvalueTy<'tcx> {
        LvalueTy::Ty { ty: ty }
    }

    pub fn to_ty(&self, tcx: &ty::ctxt<'tcx>) -> Ty<'tcx> {
        match *self {
            LvalueTy::Ty { ty } =>
                ty,
            LvalueTy::Downcast { adt_def, substs, variant_index: _ } =>
                tcx.mk_enum(adt_def, substs),
        }
    }

    pub fn projection_ty(self,
                         tcx: &ty::ctxt<'tcx>,
                         elem: &LvalueElem<'tcx>)
                         -> LvalueTy<'tcx>
    {
        match *elem {
            ProjectionElem::Deref =>
                LvalueTy::Ty {
                    ty: self.to_ty(tcx).builtin_deref(true, ty::LvaluePreference::NoPreference)
                                          .unwrap()
                                          .ty
                },
            ProjectionElem::Index(_) | ProjectionElem::ConstantIndex { .. } =>
                LvalueTy::Ty {
                    ty: self.to_ty(tcx).builtin_index().unwrap()
                },
            ProjectionElem::Downcast(adt_def1, index) =>
                match self.to_ty(tcx).sty {
                    ty::TyEnum(adt_def, substs) => {
                        assert!(index < adt_def.variants.len());
                        assert_eq!(adt_def, adt_def1);
                        LvalueTy::Downcast { adt_def: adt_def,
                                             substs: substs,
                                             variant_index: index }
                    }
                    _ => {
                        tcx.sess.bug(&format!("cannot downcast non-enum type: `{:?}`", self))
                    }
                },
            ProjectionElem::Field(field) => {
                let field_ty = match self {
                    LvalueTy::Ty { ty } => match ty.sty {
                        ty::TyStruct(adt_def, substs) =>
                            adt_def.struct_variant().fields[field.index()].ty(tcx, substs),
                        ty::TyTuple(ref tys) =>
                            tys[field.index()],
                        _ =>
                            tcx.sess.bug(&format!("cannot get field of type: `{:?}`", ty)),
                    },
                    LvalueTy::Downcast { adt_def, substs, variant_index } =>
                        adt_def.variants[variant_index].fields[field.index()].ty(tcx, substs),
                };
                LvalueTy::Ty { ty: field_ty }
            }
        }
    }
}

impl<'tcx> Mir<'tcx> {
    pub fn operand_ty(&self,
                      tcx: &ty::ctxt<'tcx>,
                      operand: &Operand<'tcx>)
                      -> Ty<'tcx>
    {
        match *operand {
            Operand::Consume(ref l) => self.lvalue_ty(tcx, l).to_ty(tcx),
            Operand::Constant(ref c) => c.ty,
        }
    }

    pub fn binop_ty(&self,
                    tcx: &ty::ctxt<'tcx>,
                    op: BinOp,
                    lhs_ty: Ty<'tcx>,
                    rhs_ty: Ty<'tcx>)
                    -> Ty<'tcx>
    {
        // FIXME: handle SIMD correctly
        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem |
            BinOp::BitXor | BinOp::BitAnd | BinOp::BitOr => {
                // these should be integers or floats of the same size.
                assert_eq!(lhs_ty, rhs_ty);
                lhs_ty
            }
            BinOp::Shl | BinOp::Shr => {
                lhs_ty // lhs_ty can be != rhs_ty
            }
            BinOp::Eq | BinOp::Lt | BinOp::Le |
            BinOp::Ne | BinOp::Ge | BinOp::Gt => {
                tcx.types.bool
            }
        }
    }

    pub fn lvalue_ty(&self,
                     tcx: &ty::ctxt<'tcx>,
                     lvalue: &Lvalue<'tcx>)
                     -> LvalueTy<'tcx>
    {
        match *lvalue {
            Lvalue::Var(index) =>
                LvalueTy::Ty { ty: self.var_decls[index as usize].ty },
            Lvalue::Temp(index) =>
                LvalueTy::Ty { ty: self.temp_decls[index as usize].ty },
            Lvalue::Arg(index) =>
                LvalueTy::Ty { ty: self.arg_decls[index as usize].ty },
            Lvalue::Static(def_id) =>
                LvalueTy::Ty { ty: tcx.lookup_item_type(def_id).ty },
            Lvalue::ReturnPointer =>
                LvalueTy::Ty { ty: self.return_ty.unwrap() },
            Lvalue::Projection(ref proj) =>
                self.lvalue_ty(tcx, &proj.base).projection_ty(tcx, &proj.elem)
        }
    }
}

impl BorrowKind {
    pub fn to_mutbl_lossy(self) -> hir::Mutability {
        match self {
            BorrowKind::Mut => hir::MutMutable,
            BorrowKind::Shared => hir::MutImmutable,

            // We have no type corresponding to a unique imm borrow, so
            // use `&mut`. It gives all the capabilities of an `&uniq`
            // and hence is a safe "over approximation".
            BorrowKind::Unique => hir::MutMutable,
        }
    }
}

impl BinOp {
    pub fn to_hir_binop(self) -> hir::BinOp_ {
        match self {
            BinOp::Add => hir::BinOp_::BiAdd,
            BinOp::Sub => hir::BinOp_::BiSub,
            BinOp::Mul => hir::BinOp_::BiMul,
            BinOp::Div => hir::BinOp_::BiDiv,
            BinOp::Rem => hir::BinOp_::BiRem,
            BinOp::BitXor => hir::BinOp_::BiBitXor,
            BinOp::BitAnd => hir::BinOp_::BiBitAnd,
            BinOp::BitOr => hir::BinOp_::BiBitOr,
            BinOp::Shl => hir::BinOp_::BiShl,
            BinOp::Shr => hir::BinOp_::BiShr,
            BinOp::Eq => hir::BinOp_::BiEq,
            BinOp::Ne => hir::BinOp_::BiNe,
            BinOp::Lt => hir::BinOp_::BiLt,
            BinOp::Gt => hir::BinOp_::BiGt,
            BinOp::Le => hir::BinOp_::BiLe,
            BinOp::Ge => hir::BinOp_::BiGe
        }
    }
}
