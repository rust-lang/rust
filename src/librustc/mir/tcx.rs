/*!
 * Methods for the various MIR types. These are intended for use after
 * building is complete.
 */

use crate::mir::*;
use crate::ty::subst::Subst;
use crate::ty::{self, Ty, TyCtxt};
use crate::ty::layout::VariantIdx;
use crate::hir;
use crate::ty::util::IntTypeExt;

#[derive(Copy, Clone, Debug)]
pub struct PlaceTy<'tcx> {
    pub ty: Ty<'tcx>,
    /// Downcast to a particular variant of an enum, if included.
    pub variant_index: Option<VariantIdx>,
}

// At least on 64 bit systems, `PlaceTy` should not be larger than two or three pointers.
#[cfg(target_arch = "x86_64")]
static_assert_size!(PlaceTy<'_>, 16);

impl<'tcx> PlaceTy<'tcx> {
    pub fn from_ty(ty: Ty<'tcx>) -> PlaceTy<'tcx> {
        PlaceTy { ty, variant_index: None }
    }

    /// `place_ty.field_ty(tcx, f)` computes the type at a given field
    /// of a record or enum-variant. (Most clients of `PlaceTy` can
    /// instead just extract the relevant type directly from their
    /// `PlaceElem`, but some instances of `ProjectionElem<V, T>` do
    /// not carry a `Ty` for `T`.)
    ///
    /// Note that the resulting type has not been normalized.
    pub fn field_ty(self, tcx: TyCtxt<'tcx>, f: &Field) -> Ty<'tcx> {
        let answer = match self.ty.sty {
            ty::Adt(adt_def, substs) => {
                let variant_def = match self.variant_index {
                    None => adt_def.non_enum_variant(),
                    Some(variant_index) => {
                        assert!(adt_def.is_enum());
                        &adt_def.variants[variant_index]
                    }
                };
                let field_def = &variant_def.fields[f.index()];
                field_def.ty(tcx, substs)
            }
            ty::Tuple(ref tys) => tys[f.index()].expect_ty(),
            _ => bug!("extracting field of non-tuple non-adt: {:?}", self),
        };
        debug!("field_ty self: {:?} f: {:?} yields: {:?}", self, f, answer);
        answer
    }

    /// Convenience wrapper around `projection_ty_core` for
    /// `PlaceElem`, where we can just use the `Ty` that is already
    /// stored inline on field projection elems.
    pub fn projection_ty(self, tcx: TyCtxt<'tcx>, elem: &PlaceElem<'tcx>) -> PlaceTy<'tcx> {
        self.projection_ty_core(tcx, elem, |_, _, ty| ty)
    }

    /// `place_ty.projection_ty_core(tcx, elem, |...| { ... })`
    /// projects `place_ty` onto `elem`, returning the appropriate
    /// `Ty` or downcast variant corresponding to that projection.
    /// The `handle_field` callback must map a `Field` to its `Ty`,
    /// (which should be trivial when `T` = `Ty`).
    pub fn projection_ty_core<V, T>(
        self,
        tcx: TyCtxt<'tcx>,
        elem: &ProjectionElem<V, T>,
        mut handle_field: impl FnMut(&Self, &Field, &T) -> Ty<'tcx>,
    ) -> PlaceTy<'tcx>
    where
        V: ::std::fmt::Debug,
        T: ::std::fmt::Debug,
    {
        let answer = match *elem {
            ProjectionElem::Deref => {
                let ty = self.ty
                             .builtin_deref(true)
                             .unwrap_or_else(|| {
                                 bug!("deref projection of non-dereferencable ty {:?}", self)
                             })
                             .ty;
                PlaceTy::from_ty(ty)
            }
            ProjectionElem::Index(_) | ProjectionElem::ConstantIndex { .. } =>
                PlaceTy::from_ty(self.ty.builtin_index().unwrap()),
            ProjectionElem::Subslice { from, to } => {
                PlaceTy::from_ty(match self.ty.sty {
                    ty::Array(inner, size) => {
                        let size = size.unwrap_usize(tcx);
                        let len = size - (from as u64) - (to as u64);
                        tcx.mk_array(inner, len)
                    }
                    ty::Slice(..) => self.ty,
                    _ => {
                        bug!("cannot subslice non-array type: `{:?}`", self)
                    }
                })
            }
            ProjectionElem::Downcast(_name, index) =>
                PlaceTy { ty: self.ty, variant_index: Some(index) },
            ProjectionElem::Field(ref f, ref fty) =>
                PlaceTy::from_ty(handle_field(&self, f, fty)),
        };
        debug!("projection_ty self: {:?} elem: {:?} yields: {:?}", self, elem, answer);
        answer
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for PlaceTy<'tcx> {
        ty,
        variant_index,
    }
}

impl<'tcx> Place<'tcx> {
    pub fn ty<D>(&self, local_decls: &D, tcx: TyCtxt<'tcx>) -> PlaceTy<'tcx>
    where
        D: HasLocalDecls<'tcx>,
    {
        self.iterate(|place_base, place_projections| {
            let mut place_ty = place_base.ty(local_decls);

            for proj in place_projections {
                place_ty = place_ty.projection_ty(tcx, &proj.elem);
            }

            place_ty
        })
    }
}

impl<'tcx> PlaceBase<'tcx> {
    pub fn ty<D>(&self, local_decls: &D) -> PlaceTy<'tcx>
        where D: HasLocalDecls<'tcx>
    {
        match self {
            PlaceBase::Local(index) => PlaceTy::from_ty(local_decls.local_decls()[*index].ty),
            PlaceBase::Static(data) => PlaceTy::from_ty(data.ty),
        }
    }
}

pub enum RvalueInitializationState {
    Shallow,
    Deep
}

impl<'tcx> Rvalue<'tcx> {
    pub fn ty<D>(&self, local_decls: &D, tcx: TyCtxt<'tcx>) -> Ty<'tcx>
    where
        D: HasLocalDecls<'tcx>,
    {
        match *self {
            Rvalue::Use(ref operand) => operand.ty(local_decls, tcx),
            Rvalue::Repeat(ref operand, count) => {
                tcx.mk_array(operand.ty(local_decls, tcx), count)
            }
            Rvalue::Ref(reg, bk, ref place) => {
                let place_ty = place.ty(local_decls, tcx).ty;
                tcx.mk_ref(reg,
                    ty::TypeAndMut {
                        ty: place_ty,
                        mutbl: bk.to_mutbl_lossy()
                    }
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
            Rvalue::UnaryOp(UnOp::Not, ref operand) |
            Rvalue::UnaryOp(UnOp::Neg, ref operand) => {
                operand.ty(local_decls, tcx)
            }
            Rvalue::Discriminant(ref place) => {
                let ty = place.ty(local_decls, tcx).ty;
                match ty.sty {
                    ty::Adt(adt_def, _) => adt_def.repr.discr_type().to_ty(tcx),
                    ty::Generator(_, substs, _) => substs.discr_ty(tcx),
                    _ => {
                        // This can only be `0`, for now, so `u8` will suffice.
                        tcx.types.u8
                    }
                }
            }
            Rvalue::NullaryOp(NullOp::Box, t) => tcx.mk_box(t),
            Rvalue::NullaryOp(NullOp::SizeOf, _) => tcx.types.usize,
            Rvalue::Aggregate(ref ak, ref ops) => {
                match **ak {
                    AggregateKind::Array(ty) => {
                        tcx.mk_array(ty, ops.len() as u64)
                    }
                    AggregateKind::Tuple => {
                        tcx.mk_tup(ops.iter().map(|op| op.ty(local_decls, tcx)))
                    }
                    AggregateKind::Adt(def, _, substs, _, _) => {
                        tcx.type_of(def.did).subst(tcx, substs)
                    }
                    AggregateKind::Closure(did, substs) => {
                        tcx.mk_closure(did, substs)
                    }
                    AggregateKind::Generator(did, substs, movability) => {
                        tcx.mk_generator(did, substs, movability)
                    }
                }
            }
        }
    }

    #[inline]
    /// Returns `true` if this rvalue is deeply initialized (most rvalues) or
    /// whether its only shallowly initialized (`Rvalue::Box`).
    pub fn initialization_state(&self) -> RvalueInitializationState {
        match *self {
            Rvalue::NullaryOp(NullOp::Box, _) => RvalueInitializationState::Shallow,
            _ => RvalueInitializationState::Deep
        }
    }
}

impl<'tcx> Operand<'tcx> {
    pub fn ty<D>(&self, local_decls: &D, tcx: TyCtxt<'tcx>) -> Ty<'tcx>
    where
        D: HasLocalDecls<'tcx>,
    {
        match self {
            &Operand::Copy(ref l) |
            &Operand::Move(ref l) => l.ty(local_decls, tcx).ty,
            &Operand::Constant(ref c) => c.ty,
        }
    }
}

impl<'tcx> BinOp {
    pub fn ty(&self, tcx: TyCtxt<'tcx>, lhs_ty: Ty<'tcx>, rhs_ty: Ty<'tcx>) -> Ty<'tcx> {
        // FIXME: handle SIMD correctly
        match self {
            &BinOp::Add | &BinOp::Sub | &BinOp::Mul | &BinOp::Div | &BinOp::Rem |
            &BinOp::BitXor | &BinOp::BitAnd | &BinOp::BitOr => {
                // these should be integers or floats of the same size.
                assert_eq!(lhs_ty, rhs_ty);
                lhs_ty
            }
            &BinOp::Shl | &BinOp::Shr | &BinOp::Offset => {
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
            BorrowKind::Mut { .. } => hir::MutMutable,
            BorrowKind::Shared => hir::MutImmutable,

            // We have no type corresponding to a unique imm borrow, so
            // use `&mut`. It gives all the capabilities of an `&uniq`
            // and hence is a safe "over approximation".
            BorrowKind::Unique => hir::MutMutable,

            // We have no type corresponding to a shallow borrow, so use
            // `&` as an approximation.
            BorrowKind::Shallow => hir::MutImmutable,
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
            BinOp::Offset => unreachable!()
        }
    }
}
