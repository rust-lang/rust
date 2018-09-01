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

use hir;
use mir::*;
use ty::subst::{Subst, Substs};
use ty::util::IntTypeExt;
use ty::{self, AdtDef, Ty, TyCtxt};

#[derive(Copy, Clone, Debug)]
pub enum PlaceTy<'tcx> {
    /// Normal type.
    Ty { ty: Ty<'tcx> },

    /// Downcast to a particular variant of an enum.
    Downcast {
        adt_def: &'tcx AdtDef,
        substs: &'tcx Substs<'tcx>,
        variant_index: usize,
    },
}

impl<'a, 'gcx, 'tcx> PlaceTy<'tcx> {
    pub fn to_ty(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Ty<'tcx> {
        match *self {
            PlaceTy::Ty { ty } => ty,
            PlaceTy::Downcast {
                adt_def,
                substs,
                variant_index: _,
            } => tcx.mk_adt(adt_def, substs),
        }
    }

    pub fn projection_ty(
        self,
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        elem: &PlaceElem<'tcx>,
    ) -> PlaceTy<'tcx> {
        match elem {
            ProjectionElem::Deref => {
                let ty = self
                    .to_ty(tcx)
                    .builtin_deref(true)
                    .unwrap_or_else(|| bug!("deref projection of non-dereferencable ty {:?}", self))
                    .ty;
                PlaceTy::Ty { ty }
            }
            ProjectionElem::Index(_) | ProjectionElem::ConstantIndex { .. } => PlaceTy::Ty {
                ty: self.to_ty(tcx).builtin_index().unwrap(),
            },
            ProjectionElem::Subslice { from, to } => {
                let ty = self.to_ty(tcx);
                PlaceTy::Ty {
                    ty: match ty.sty {
                        ty::TyArray(inner, size) => {
                            let size = size.unwrap_usize(tcx);
                            let len = size - (*from as u64) - (*to as u64);
                            tcx.mk_array(inner, len)
                        }
                        ty::TySlice(..) => ty,
                        _ => bug!("cannot subslice non-array type: `{:?}`", self),
                    },
                }
            }
            ProjectionElem::Downcast(adt_def1, index) => match self.to_ty(tcx).sty {
                ty::TyAdt(adt_def, substs) => {
                    assert!(adt_def.is_enum());
                    assert!(*index < adt_def.variants.len());
                    assert_eq!(adt_def, *adt_def1);
                    PlaceTy::Downcast {
                        adt_def,
                        substs,
                        variant_index: *index,
                    }
                }
                _ => bug!("cannot downcast non-ADT type: `{:?}`", self),
            },
            ProjectionElem::Field(_, fty) => PlaceTy::Ty { ty: fty },
        }
    }
}

impl From<Ty<'tcx>> for PlaceTy<'tcx> {
    fn from(ty: Ty<'tcx>) -> Self {
        PlaceTy::Ty { ty }
    }
}

EnumTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for PlaceTy<'tcx> {
        (PlaceTy::Ty) { ty },
        (PlaceTy::Downcast) { adt_def, substs, variant_index },
    }
}

impl<'tcx> PlaceBase<'tcx> {
    pub fn ty(&self, local_decls: &impl HasLocalDecls<'tcx>) -> Ty<'tcx> {
        match self {
            PlaceBase::Local(index) => local_decls.local_decls()[*index].ty,
            PlaceBase::Promoted(data) => data.1,
            PlaceBase::Static(data) => data.ty,
        }
    }
}

impl<'tcx> Place<'tcx> {
    pub fn ty<'a, 'gcx>(
        &self,
        local_decls: &impl HasLocalDecls<'tcx>,
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
    ) -> PlaceTy<'tcx> {
        // the PlaceTy is the *final* type with all projection applied
        // if there is no projection, that just refers to `base`:
        //
        // Place: base.[a, b, c]
        //                    ^-- projection
        //                    ^-- PlaceTy
        //
        // Place: base.[]
        //             ^^-- no projection
        //        ^^^^-- PlaceTy

        let mut place_ty = PlaceTy::from(self.base.ty(local_decls));

        // apply .projection_ty() to all elems but only returns the final one.
        for elem in self.elems.iter() {
            place_ty = place_ty.projection_ty(tcx, elem);
        }

        place_ty
    }

    /// If this is a field projection, and the field is being projected from a closure type,
    /// then returns the index of the field being projected. Note that this closure will always
    /// be `self` in the current MIR, because that is the only time we directly access the fields
    /// of a closure type.
    pub fn is_upvar_field_projection<'cx, 'gcx>(
        &self,
        mir: &'cx Mir<'tcx>,
        tcx: &TyCtxt<'cx, 'gcx, 'tcx>,
    ) -> Option<Field> {
        let base_place;
        let mut place = self;
        let mut by_ref = false;

        base_place = place.projection_base(*tcx);

        if let Some(ProjectionElem::Deref) = place.projection() {
            place = &base_place;
            by_ref = true;
        }
        if let Some(ProjectionElem::Field(field, _ty)) = place.projection() {
            let base_ty = place.projection_base(*tcx).ty(mir, *tcx).to_ty(*tcx);

            if base_ty.is_closure()
                || base_ty.is_generator() && (!(by_ref && !mir.upvar_decls[field.index()].by_ref))
            {
                Some(*field)
            } else {
                None
            }
        } else {
            None
        }
    }

    // for Place:
    //    (Base.[a, b, c])
    //     ^^^^^^^^^^  ^-- projection
    //     |-- base_place
    //
    //     Base.[]
    //     ^^^^ ^^-- no projection(empty)
    //     |-- base_place
    pub fn split_projection<'cx, 'gcx>(
        &self,
        tcx: TyCtxt<'cx, 'gcx, 'tcx>,
    ) -> (Place<'tcx>, Option<&'tcx PlaceElem<'tcx>>) {
        // split place_elems
        // Base.[a, b, c]
        //       ^^^^  ^-- projection(projection lives in the last elem)
        //       |-- place_elems
        match self.elems.split_last() {
            Some((projection, place_elems)) => (
                Place {
                    base: self.clone().base,
                    elems: tcx.intern_place_elems(place_elems),
                },
                Some(projection),
            ),
            _ => (self.clone(), None)
        }
    }

    pub fn has_no_projection(&self) -> bool {
        self.elems.is_empty()
    }

    // for projection returns the base place;
    //     Base.[a, b, c] => Base.[a, b]
    //                 ^-- projection
    // if no projection returns the place itself,
    //     Base.[] => Base.[]
    //          ^^-- no projection
    pub fn projection_base<'cx, 'gcx>(&self, tcx: TyCtxt<'cx, 'gcx, 'tcx>) -> Place<'tcx> {
        match self.split_projection(tcx) {
            (place, Some(_)) => place,
            (_, None) => self.clone(),
        }
    }

    // for a place_elem returns it's base projection
    // Base.[a, b, c]
    //          ^-- place_elem
    // ^^^^^^^-- base
    pub fn elem_base<'cx, 'gcx>(
        &self,
        tcx: TyCtxt<'cx, 'gcx, 'tcx>,
        elem_index: usize,
    ) -> Place<'tcx> {
        // only works for place with projections
        assert!(!self.has_no_projection());

        if elem_index < 1 {
            // Base.[a]
            //       ^-- elems[0]
            Place {
                base: self.clone().base,
                elems: Slice::empty(),
            }
        } else {
            Place {
                base: self.clone().base,
                elems: tcx.mk_place_elems(
                    self.elems.iter().cloned().take(elem_index)
                )
            }
        }
    }
}

pub enum RvalueInitializationState {
    Shallow,
    Deep,
}

impl<'tcx> Rvalue<'tcx> {
    pub fn ty<'a, 'gcx, D>(&self, local_decls: &D, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Ty<'tcx>
    where
        D: HasLocalDecls<'tcx>,
    {
        match *self {
            Rvalue::Use(ref operand) => operand.ty(local_decls, tcx),
            Rvalue::Repeat(ref operand, count) => tcx.mk_array(operand.ty(local_decls, tcx), count),
            Rvalue::Ref(reg, bk, ref place) => {
                let place_ty = place.ty(local_decls, tcx).to_ty(tcx);
                tcx.mk_ref(
                    reg,
                    ty::TypeAndMut {
                        ty: place_ty,
                        mutbl: bk.to_mutbl_lossy(),
                    },
                )
            }
            Rvalue::Len(..) => tcx.types.usize,
            Rvalue::Cast(.., ty) => ty,
            Rvalue::BinaryOp(op, ref lhs, ref rhs) => {
                let lhs_ty = lhs.ty(local_decls, tcx);
                let rhs_ty = rhs.ty(local_decls, tcx);
                op.ty(tcx, lhs_ty, rhs_ty)
            }
            Rvalue::CheckedBinaryOp(op, ref lhs, ref rhs) => {
                let lhs_ty = lhs.ty(local_decls, tcx);
                let rhs_ty = rhs.ty(local_decls, tcx);
                let ty = op.ty(tcx, lhs_ty, rhs_ty);
                tcx.intern_tup(&[ty, tcx.types.bool])
            }
            Rvalue::UnaryOp(UnOp::Not, ref operand) | Rvalue::UnaryOp(UnOp::Neg, ref operand) => {
                operand.ty(local_decls, tcx)
            }
            Rvalue::Discriminant(ref place) => {
                let ty = place.ty(local_decls, tcx).to_ty(tcx);
                if let ty::TyAdt(adt_def, _) = ty.sty {
                    adt_def.repr.discr_type().to_ty(tcx)
                } else {
                    // This can only be `0`, for now, so `u8` will suffice.
                    tcx.types.u8
                }
            }
            Rvalue::NullaryOp(NullOp::Box, t) => tcx.mk_box(t),
            Rvalue::NullaryOp(NullOp::SizeOf, _) => tcx.types.usize,
            Rvalue::Aggregate(ref ak, ref ops) => match **ak {
                AggregateKind::Array(ty) => tcx.mk_array(ty, ops.len() as u64),
                AggregateKind::Tuple => tcx.mk_tup(ops.iter().map(|op| op.ty(local_decls, tcx))),
                AggregateKind::Adt(def, _, substs, _) => tcx.type_of(def.did).subst(tcx, substs),
                AggregateKind::Closure(did, substs) => tcx.mk_closure(did, substs),
                AggregateKind::Generator(did, substs, movability) => {
                    tcx.mk_generator(did, substs, movability)
                }
            },
        }
    }

    #[inline]
    /// Returns whether this rvalue is deeply initialized (most rvalues) or
    /// whether its only shallowly initialized (`Rvalue::Box`).
    pub fn initialization_state(&self) -> RvalueInitializationState {
        match *self {
            Rvalue::NullaryOp(NullOp::Box, _) => RvalueInitializationState::Shallow,
            _ => RvalueInitializationState::Deep,
        }
    }
}

impl<'tcx> Operand<'tcx> {
    pub fn ty<'a, 'gcx, D>(&self, local_decls: &D, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Ty<'tcx>
    where
        D: HasLocalDecls<'tcx>,
    {
        match self {
            &Operand::Copy(ref l) | &Operand::Move(ref l) => l.ty(local_decls, tcx).to_ty(tcx),
            &Operand::Constant(ref c) => c.ty,
        }
    }
}

impl<'tcx> BinOp {
    pub fn ty<'a, 'gcx>(
        &self,
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        lhs_ty: Ty<'tcx>,
        rhs_ty: Ty<'tcx>,
    ) -> Ty<'tcx> {
        // FIXME: handle SIMD correctly
        match self {
            &BinOp::Add
            | &BinOp::Sub
            | &BinOp::Mul
            | &BinOp::Div
            | &BinOp::Rem
            | &BinOp::BitXor
            | &BinOp::BitAnd
            | &BinOp::BitOr => {
                // these should be integers or floats of the same size.
                assert_eq!(lhs_ty, rhs_ty);
                lhs_ty
            }
            &BinOp::Shl | &BinOp::Shr | &BinOp::Offset => {
                lhs_ty // lhs_ty can be != rhs_ty
            }
            &BinOp::Eq | &BinOp::Lt | &BinOp::Le | &BinOp::Ne | &BinOp::Ge | &BinOp::Gt => {
                tcx.types.bool
            }
        }
    }
}

impl BorrowKind {
    pub fn to_mutbl_lossy(self) -> hir::Mutability {
        match self {
            BorrowKind::Mut { .. } => hir::MutMutable,
            BorrowKind::Shared => hir::MutImmutable,

            // We have no type corresponding to a unique imm borrow, so
            // use `&mut`. It gives all the capabilities of an `&uniq`
            // and hence is a safe "over approximation".
            BorrowKind::Unique => hir::MutMutable,
        }
    }
}

impl BinOp {
    pub fn to_hir_binop(self) -> hir::BinOpKind {
        match self {
            BinOp::Add => hir::BinOpKind::Add,
            BinOp::Sub => hir::BinOpKind::Sub,
            BinOp::Mul => hir::BinOpKind::Mul,
            BinOp::Div => hir::BinOpKind::Div,
            BinOp::Rem => hir::BinOpKind::Rem,
            BinOp::BitXor => hir::BinOpKind::BitXor,
            BinOp::BitAnd => hir::BinOpKind::BitAnd,
            BinOp::BitOr => hir::BinOpKind::BitOr,
            BinOp::Shl => hir::BinOpKind::Shl,
            BinOp::Shr => hir::BinOpKind::Shr,
            BinOp::Eq => hir::BinOpKind::Eq,
            BinOp::Ne => hir::BinOpKind::Ne,
            BinOp::Lt => hir::BinOpKind::Lt,
            BinOp::Gt => hir::BinOpKind::Gt,
            BinOp::Le => hir::BinOpKind::Le,
            BinOp::Ge => hir::BinOpKind::Ge,
            BinOp::Offset => unreachable!(),
        }
    }
}
