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

use mir::repr::*;
use ty::subst::{Subst, Substs};
use ty::{self, AdtDef, Ty, TyCtxt};
use ty::fold::{TypeFoldable, TypeFolder, TypeVisitor};
use hir;

#[derive(Copy, Clone, Debug)]
pub enum LvalueTy<'tcx> {
    /// Normal type.
    Ty { ty: Ty<'tcx> },

    /// Downcast to a particular variant of an enum.
    Downcast { adt_def: AdtDef<'tcx>,
               substs: &'tcx Substs<'tcx>,
               variant_index: usize },
}

impl<'a, 'gcx, 'tcx> LvalueTy<'tcx> {
    pub fn from_ty(ty: Ty<'tcx>) -> LvalueTy<'tcx> {
        LvalueTy::Ty { ty: ty }
    }

    pub fn to_ty(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Ty<'tcx> {
        match *self {
            LvalueTy::Ty { ty } =>
                ty,
            LvalueTy::Downcast { adt_def, substs, variant_index: _ } =>
                tcx.mk_enum(adt_def, substs),
        }
    }

    pub fn projection_ty(self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
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
            ProjectionElem::Subslice { from, to } => {
                let ty = self.to_ty(tcx);
                LvalueTy::Ty {
                    ty: match ty.sty {
                        ty::TyArray(inner, size) => {
                            tcx.mk_array(inner, size-(from as usize)-(to as usize))
                        }
                        ty::TySlice(..) => ty,
                        _ => {
                            bug!("cannot subslice non-array type: `{:?}`", self)
                        }
                    }
                }
            }
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
                        bug!("cannot downcast non-enum type: `{:?}` as `{:?}`", self, elem)
                    }
                },
            ProjectionElem::Field(_, fty) => LvalueTy::Ty { ty: fty }
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for LvalueTy<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        match *self {
            LvalueTy::Ty { ty } => LvalueTy::Ty { ty: ty.fold_with(folder) },
            LvalueTy::Downcast { adt_def, substs, variant_index } => {
                LvalueTy::Downcast {
                    adt_def: adt_def,
                    substs: substs.fold_with(folder),
                    variant_index: variant_index
                }
            }
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        match *self {
            LvalueTy::Ty { ty } => ty.visit_with(visitor),
            LvalueTy::Downcast { substs, .. } => substs.visit_with(visitor)
        }
    }
}

impl<'a, 'gcx, 'tcx> Mir<'tcx> {
    pub fn operand_ty(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                      operand: &Operand<'tcx>)
                      -> Ty<'tcx>
    {
        match *operand {
            Operand::Consume(ref l) => self.lvalue_ty(tcx, l).to_ty(tcx),
            Operand::Constant(ref c) => c.ty,
        }
    }

    pub fn ty(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>, lvalue: &Lvalue<'tcx>) -> LvalueTy<'tcx> {
        match *lvalue {
            Lvalue::Var(index) =>
                LvalueTy::Ty { ty: self.var_decls[index].ty },
            Lvalue::Temp(index) =>
                LvalueTy::Ty { ty: self.temp_decls[index].ty },
            Lvalue::Arg(index) =>
                LvalueTy::Ty { ty: self.arg_decls[index].ty },
            Lvalue::Static(def_id) =>
                LvalueTy::Ty { ty: tcx.lookup_item_type(def_id).ty },
            &Lvalue::ReturnPointer =>
                LvalueTy::Ty { ty: mir.return_ty },
            &Lvalue::Projection(ref proj) =>
                proj.base.ty(mir, tcx).projection_ty(tcx, &proj.elem),
        }
    }
}

impl<'tcx> Rvalue<'tcx> {
    pub fn ty<'a, 'gcx>(&self, mir: &Mir<'tcx>, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Option<Ty<'tcx>>
    {
        match self {
            &Rvalue::Use(ref operand) => Some(operand.ty(mir, tcx)),
            &Rvalue::Repeat(ref operand, ref count) => {
                let op_ty = operand.ty(mir, tcx);
                let count = count.value.as_u64(tcx.sess.target.uint_type);
                assert_eq!(count as usize as u64, count);
                Some(tcx.mk_array(op_ty, count as usize))
            }
            &Rvalue::Ref(reg, bk, ref lv) => {
                let lv_ty = lv.ty(mir, tcx).to_ty(tcx);
                Some(tcx.mk_ref(
                    tcx.mk_region(reg),
                    ty::TypeAndMut {
                        ty: lv_ty,
                        mutbl: bk.to_mutbl_lossy()
                    }
                ))
            }
            Rvalue::Len(..) => Some(tcx.types.usize),
            Rvalue::Cast(_, _, ty) => Some(ty),
            Rvalue::BinaryOp(op, ref lhs, ref rhs) => {
                let lhs_ty = self.operand_ty(tcx, lhs);
                let rhs_ty = self.operand_ty(tcx, rhs);
                Some(binop_ty(tcx, op, lhs_ty, rhs_ty))
            }
            Rvalue::CheckedBinaryOp(op, ref lhs, ref rhs) => {
                let lhs_ty = self.operand_ty(tcx, lhs);
                let rhs_ty = self.operand_ty(tcx, rhs);
                let ty = binop_ty(tcx, op, lhs_ty, rhs_ty);
                let ty = tcx.mk_tup(vec![ty, tcx.types.bool]);
                Some(ty)
            }
            &Rvalue::UnaryOp(_, ref operand) => {
                Some(operand.ty(mir, tcx))
            }
            &Rvalue::Box(t) => {
                Some(tcx.mk_box(t))
            }
            &Rvalue::Aggregate(ref ak, ref ops) => {
                match *ak {
                    AggregateKind::Vec => {
                        if let Some(operand) = ops.get(0) {
                            let ty = operand.ty(mir, tcx);
                            Some(tcx.mk_array(ty, ops.len()))
                        } else {
                            None
                        }
                    }
                    AggregateKind::Tuple => {
                        Some(tcx.mk_tup(
                            ops.iter().map(|op| op.ty(mir, tcx)).collect()
                        ))
                    }
                    AggregateKind::Adt(def, _, substs) => {
                        Some(tcx.lookup_item_type(def.did).ty.subst(tcx, substs))
                    }
                    AggregateKind::Closure(did, substs) => {
                        Some(tcx.mk_closure_from_closure_substs(did, substs))
                    }
                }
            }
            &Rvalue::InlineAsm { .. } => None
        }
    }
}

impl<'tcx> Operand<'tcx> {
    pub fn ty<'a, 'gcx>(&self, mir: &Mir<'tcx>, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Ty<'tcx> {
        match self {
            &Operand::Consume(ref l) => l.ty(mir, tcx).to_ty(tcx),
            &Operand::Constant(ref c) => c.ty,
        }
    }
}

impl<'tcx> BinOp {
      pub fn ty<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                    lhs_ty: Ty<'tcx>,
                    rhs_ty: Ty<'tcx>)
                    -> Ty<'tcx> {
        // FIXME: handle SIMD correctly
        match self {
            &BinOp::Add | &BinOp::Sub | &BinOp::Mul | &BinOp::Div | &BinOp::Rem |
            &BinOp::BitXor | &BinOp::BitAnd | &BinOp::BitOr => {
                // these should be integers or floats of the same size.
                assert_eq!(lhs_ty, rhs_ty);
                lhs_ty
            }
            &BinOp::Shl | &BinOp::Shr => {
                lhs_ty // lhs_ty can be != rhs_ty
            }
            &BinOp::Eq | &BinOp::Lt | &BinOp::Le |
            &BinOp::Ne | &BinOp::Ge | &BinOp::Gt => {
                tcx.types.bool
            }
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
