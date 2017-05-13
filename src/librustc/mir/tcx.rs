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

use mir::*;
use ty::subst::{Subst, Substs};
use ty::{self, AdtDef, Ty, TyCtxt};
use ty::fold::{TypeFoldable, TypeFolder, TypeVisitor};
use hir;

#[derive(Copy, Clone, Debug)]
pub enum LvalueTy<'tcx> {
    /// Normal type.
    Ty { ty: Ty<'tcx> },

    /// Downcast to a particular variant of an enum.
    Downcast { adt_def: &'tcx AdtDef,
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
                tcx.mk_adt(adt_def, substs),
        }
    }

    pub fn projection_ty(self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                         elem: &LvalueElem<'tcx>)
                         -> LvalueTy<'tcx>
    {
        match *elem {
            ProjectionElem::Deref => {
                let ty = self.to_ty(tcx)
                             .builtin_deref(true, ty::LvaluePreference::NoPreference)
                             .unwrap_or_else(|| {
                                 bug!("deref projection of non-dereferencable ty {:?}", self)
                             })
                             .ty;
                LvalueTy::Ty {
                    ty: ty,
                }
            }
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
                    ty::TyAdt(adt_def, substs) => {
                        assert!(adt_def.is_enum());
                        assert!(index < adt_def.variants.len());
                        assert_eq!(adt_def, adt_def1);
                        LvalueTy::Downcast { adt_def: adt_def,
                                             substs: substs,
                                             variant_index: index }
                    }
                    _ => {
                        bug!("cannot downcast non-ADT type: `{:?}`", self)
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

impl<'tcx> Lvalue<'tcx> {
    pub fn ty<'a, 'gcx>(&self, mir: &Mir<'tcx>, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> LvalueTy<'tcx> {
        match *self {
            Lvalue::Local(index) =>
                LvalueTy::Ty { ty: mir.local_decls[index].ty },
            Lvalue::Static(def_id) =>
                LvalueTy::Ty { ty: tcx.item_type(def_id) },
            Lvalue::Projection(ref proj) =>
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
                Some(tcx.mk_ref(reg,
                    ty::TypeAndMut {
                        ty: lv_ty,
                        mutbl: bk.to_mutbl_lossy()
                    }
                ))
            }
            &Rvalue::Len(..) => Some(tcx.types.usize),
            &Rvalue::Cast(.., ty) => Some(ty),
            &Rvalue::BinaryOp(op, ref lhs, ref rhs) => {
                let lhs_ty = lhs.ty(mir, tcx);
                let rhs_ty = rhs.ty(mir, tcx);
                Some(op.ty(tcx, lhs_ty, rhs_ty))
            }
            &Rvalue::CheckedBinaryOp(op, ref lhs, ref rhs) => {
                let lhs_ty = lhs.ty(mir, tcx);
                let rhs_ty = rhs.ty(mir, tcx);
                let ty = op.ty(tcx, lhs_ty, rhs_ty);
                let ty = tcx.intern_tup(&[ty, tcx.types.bool]);
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
                    AggregateKind::Array => {
                        if let Some(operand) = ops.get(0) {
                            let ty = operand.ty(mir, tcx);
                            Some(tcx.mk_array(ty, ops.len()))
                        } else {
                            None
                        }
                    }
                    AggregateKind::Tuple => {
                        Some(tcx.mk_tup(
                            ops.iter().map(|op| op.ty(mir, tcx))
                        ))
                    }
                    AggregateKind::Adt(def, _, substs, _) => {
                        Some(tcx.item_type(def.did).subst(tcx, substs))
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
