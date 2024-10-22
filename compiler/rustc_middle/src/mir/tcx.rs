/*!
 * Methods for the various MIR types. These are intended for use after
 * building is complete.
 */

use rustc_hir as hir;
use tracing::{debug, instrument};

use crate::mir::*;

#[derive(Copy, Clone, Debug, TypeFoldable, TypeVisitable)]
pub struct PlaceTy<'tcx> {
    pub ty: Ty<'tcx>,
    /// Downcast to a particular variant of an enum or a coroutine, if included.
    pub variant_index: Option<VariantIdx>,
}

// At least on 64 bit systems, `PlaceTy` should not be larger than two or three pointers.
#[cfg(target_pointer_width = "64")]
rustc_data_structures::static_assert_size!(PlaceTy<'_>, 16);

impl<'tcx> PlaceTy<'tcx> {
    #[inline]
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
    #[instrument(level = "debug", skip(tcx), ret)]
    pub fn field_ty(self, tcx: TyCtxt<'tcx>, f: FieldIdx) -> Ty<'tcx> {
        match self.ty.kind() {
            ty::Adt(adt_def, args) => {
                let variant_def = match self.variant_index {
                    None => adt_def.non_enum_variant(),
                    Some(variant_index) => {
                        assert!(adt_def.is_enum());
                        adt_def.variant(variant_index)
                    }
                };
                let field_def = &variant_def.fields[f];
                field_def.ty(tcx, args)
            }
            ty::Tuple(tys) => tys[f.index()],
            _ => bug!("extracting field of non-tuple non-adt: {:?}", self),
        }
    }

    /// Convenience wrapper around `projection_ty_core` for
    /// `PlaceElem`, where we can just use the `Ty` that is already
    /// stored inline on field projection elems.
    pub fn projection_ty(self, tcx: TyCtxt<'tcx>, elem: PlaceElem<'tcx>) -> PlaceTy<'tcx> {
        self.projection_ty_core(tcx, &elem, |_, _, ty| ty, |_, ty| ty)
    }

    /// `place_ty.projection_ty_core(tcx, elem, |...| { ... })`
    /// projects `place_ty` onto `elem`, returning the appropriate
    /// `Ty` or downcast variant corresponding to that projection.
    /// The `handle_field` callback must map a `FieldIdx` to its `Ty`,
    /// (which should be trivial when `T` = `Ty`).
    pub fn projection_ty_core<V, T>(
        self,
        tcx: TyCtxt<'tcx>,
        elem: &ProjectionElem<V, T>,
        mut handle_field: impl FnMut(&Self, FieldIdx, T) -> Ty<'tcx>,
        mut handle_opaque_cast_and_subtype: impl FnMut(&Self, T) -> Ty<'tcx>,
    ) -> PlaceTy<'tcx>
    where
        V: ::std::fmt::Debug,
        T: ::std::fmt::Debug + Copy,
    {
        if self.variant_index.is_some() && !matches!(elem, ProjectionElem::Field(..)) {
            bug!("cannot use non field projection on downcasted place")
        }
        let answer = match *elem {
            ProjectionElem::Deref => {
                let ty = self.ty.builtin_deref(true).unwrap_or_else(|| {
                    bug!("deref projection of non-dereferenceable ty {:?}", self)
                });
                PlaceTy::from_ty(ty)
            }
            ProjectionElem::Index(_) | ProjectionElem::ConstantIndex { .. } => {
                PlaceTy::from_ty(self.ty.builtin_index().unwrap())
            }
            ProjectionElem::Subslice { from, to, from_end } => {
                PlaceTy::from_ty(match self.ty.kind() {
                    ty::Slice(..) => self.ty,
                    ty::Array(inner, _) if !from_end => Ty::new_array(tcx, *inner, to - from),
                    ty::Array(inner, size) if from_end => {
                        let size = size
                            .try_to_target_usize(tcx)
                            .expect("expected subslice projection on fixed-size array");
                        let len = size - from - to;
                        Ty::new_array(tcx, *inner, len)
                    }
                    _ => bug!("cannot subslice non-array type: `{:?}`", self),
                })
            }
            ProjectionElem::Downcast(_name, index) => {
                PlaceTy { ty: self.ty, variant_index: Some(index) }
            }
            ProjectionElem::Field(f, fty) => PlaceTy::from_ty(handle_field(&self, f, fty)),
            ProjectionElem::OpaqueCast(ty) => {
                PlaceTy::from_ty(handle_opaque_cast_and_subtype(&self, ty))
            }
            ProjectionElem::Subtype(ty) => {
                PlaceTy::from_ty(handle_opaque_cast_and_subtype(&self, ty))
            }
        };
        debug!("projection_ty self: {:?} elem: {:?} yields: {:?}", self, elem, answer);
        answer
    }
}

impl<'tcx> Place<'tcx> {
    pub fn ty_from<D: ?Sized>(
        local: Local,
        projection: &[PlaceElem<'tcx>],
        local_decls: &D,
        tcx: TyCtxt<'tcx>,
    ) -> PlaceTy<'tcx>
    where
        D: HasLocalDecls<'tcx>,
    {
        projection
            .iter()
            .fold(PlaceTy::from_ty(local_decls.local_decls()[local].ty), |place_ty, &elem| {
                place_ty.projection_ty(tcx, elem)
            })
    }

    pub fn ty<D: ?Sized>(&self, local_decls: &D, tcx: TyCtxt<'tcx>) -> PlaceTy<'tcx>
    where
        D: HasLocalDecls<'tcx>,
    {
        Place::ty_from(self.local, self.projection, local_decls, tcx)
    }
}

impl<'tcx> PlaceRef<'tcx> {
    pub fn ty<D: ?Sized>(&self, local_decls: &D, tcx: TyCtxt<'tcx>) -> PlaceTy<'tcx>
    where
        D: HasLocalDecls<'tcx>,
    {
        Place::ty_from(self.local, self.projection, local_decls, tcx)
    }
}

pub enum RvalueInitializationState {
    Shallow,
    Deep,
}

impl<'tcx> Rvalue<'tcx> {
    pub fn ty<D: ?Sized>(&self, local_decls: &D, tcx: TyCtxt<'tcx>) -> Ty<'tcx>
    where
        D: HasLocalDecls<'tcx>,
    {
        match *self {
            Rvalue::Use(ref operand) => operand.ty(local_decls, tcx),
            Rvalue::Repeat(ref operand, count) => {
                Ty::new_array_with_const_len(tcx, operand.ty(local_decls, tcx), count)
            }
            Rvalue::ThreadLocalRef(did) => tcx.thread_local_ptr_ty(did),
            Rvalue::Ref(reg, bk, ref place) => {
                let place_ty = place.ty(local_decls, tcx).ty;
                Ty::new_ref(tcx, reg, place_ty, bk.to_mutbl_lossy())
            }
            Rvalue::RawPtr(mutability, ref place) => {
                let place_ty = place.ty(local_decls, tcx).ty;
                Ty::new_ptr(tcx, place_ty, mutability)
            }
            Rvalue::Len(..) => tcx.types.usize,
            Rvalue::Cast(.., ty) => ty,
            Rvalue::BinaryOp(op, box (ref lhs, ref rhs)) => {
                let lhs_ty = lhs.ty(local_decls, tcx);
                let rhs_ty = rhs.ty(local_decls, tcx);
                op.ty(tcx, lhs_ty, rhs_ty)
            }
            Rvalue::UnaryOp(op, ref operand) => {
                let arg_ty = operand.ty(local_decls, tcx);
                op.ty(tcx, arg_ty)
            }
            Rvalue::Discriminant(ref place) => place.ty(local_decls, tcx).ty.discriminant_ty(tcx),
            Rvalue::NullaryOp(NullOp::SizeOf | NullOp::AlignOf | NullOp::OffsetOf(..), _) => {
                tcx.types.usize
            }
            Rvalue::NullaryOp(NullOp::UbChecks, _) => tcx.types.bool,
            Rvalue::Aggregate(ref ak, ref ops) => match **ak {
                AggregateKind::Array(ty) => Ty::new_array(tcx, ty, ops.len() as u64),
                AggregateKind::Tuple => {
                    Ty::new_tup_from_iter(tcx, ops.iter().map(|op| op.ty(local_decls, tcx)))
                }
                AggregateKind::Adt(did, _, args, _, _) => tcx.type_of(did).instantiate(tcx, args),
                AggregateKind::Closure(did, args) => Ty::new_closure(tcx, did, args),
                AggregateKind::Coroutine(did, args) => Ty::new_coroutine(tcx, did, args),
                AggregateKind::CoroutineClosure(did, args) => {
                    Ty::new_coroutine_closure(tcx, did, args)
                }
                AggregateKind::RawPtr(ty, mutability) => Ty::new_ptr(tcx, ty, mutability),
            },
            Rvalue::ShallowInitBox(_, ty) => Ty::new_box(tcx, ty),
            Rvalue::CopyForDeref(ref place) => place.ty(local_decls, tcx).ty,
        }
    }

    #[inline]
    /// Returns `true` if this rvalue is deeply initialized (most rvalues) or
    /// whether its only shallowly initialized (`Rvalue::Box`).
    pub fn initialization_state(&self) -> RvalueInitializationState {
        match *self {
            Rvalue::ShallowInitBox(_, _) => RvalueInitializationState::Shallow,
            _ => RvalueInitializationState::Deep,
        }
    }
}

impl<'tcx> Operand<'tcx> {
    pub fn ty<D: ?Sized>(&self, local_decls: &D, tcx: TyCtxt<'tcx>) -> Ty<'tcx>
    where
        D: HasLocalDecls<'tcx>,
    {
        match self {
            &Operand::Copy(ref l) | &Operand::Move(ref l) => l.ty(local_decls, tcx).ty,
            Operand::Constant(c) => c.const_.ty(),
        }
    }

    pub fn span<D: ?Sized>(&self, local_decls: &D) -> Span
    where
        D: HasLocalDecls<'tcx>,
    {
        match self {
            &Operand::Copy(ref l) | &Operand::Move(ref l) => {
                local_decls.local_decls()[l.local].source_info.span
            }
            Operand::Constant(c) => c.span,
        }
    }
}

impl<'tcx> BinOp {
    pub fn ty(&self, tcx: TyCtxt<'tcx>, lhs_ty: Ty<'tcx>, rhs_ty: Ty<'tcx>) -> Ty<'tcx> {
        // FIXME: handle SIMD correctly
        match self {
            &BinOp::Add
            | &BinOp::AddUnchecked
            | &BinOp::Sub
            | &BinOp::SubUnchecked
            | &BinOp::Mul
            | &BinOp::MulUnchecked
            | &BinOp::Div
            | &BinOp::Rem
            | &BinOp::BitXor
            | &BinOp::BitAnd
            | &BinOp::BitOr => {
                // these should be integers or floats of the same size.
                assert_eq!(lhs_ty, rhs_ty);
                lhs_ty
            }
            &BinOp::AddWithOverflow | &BinOp::SubWithOverflow | &BinOp::MulWithOverflow => {
                // these should be integers of the same size.
                assert_eq!(lhs_ty, rhs_ty);
                Ty::new_tup(tcx, &[lhs_ty, tcx.types.bool])
            }
            &BinOp::Shl
            | &BinOp::ShlUnchecked
            | &BinOp::Shr
            | &BinOp::ShrUnchecked
            | &BinOp::Offset => {
                lhs_ty // lhs_ty can be != rhs_ty
            }
            &BinOp::Eq | &BinOp::Lt | &BinOp::Le | &BinOp::Ne | &BinOp::Ge | &BinOp::Gt => {
                tcx.types.bool
            }
            &BinOp::Cmp => {
                // these should be integer-like types of the same size.
                assert_eq!(lhs_ty, rhs_ty);
                tcx.ty_ordering_enum(None)
            }
        }
    }
}

impl<'tcx> UnOp {
    pub fn ty(&self, tcx: TyCtxt<'tcx>, arg_ty: Ty<'tcx>) -> Ty<'tcx> {
        match self {
            UnOp::Not | UnOp::Neg => arg_ty,
            UnOp::PtrMetadata => arg_ty.pointee_metadata_ty_or_projection(tcx),
        }
    }
}

impl BorrowKind {
    pub fn to_mutbl_lossy(self) -> hir::Mutability {
        match self {
            BorrowKind::Mut { .. } => hir::Mutability::Mut,
            BorrowKind::Shared => hir::Mutability::Not,

            // We have no type corresponding to a shallow borrow, so use
            // `&` as an approximation.
            BorrowKind::Fake(_) => hir::Mutability::Not,
        }
    }
}

impl BinOp {
    pub(crate) fn to_hir_binop(self) -> hir::BinOpKind {
        match self {
            // HIR `+`/`-`/`*` can map to either of these MIR BinOp, depending
            // on whether overflow checks are enabled or not.
            BinOp::Add | BinOp::AddWithOverflow => hir::BinOpKind::Add,
            BinOp::Sub | BinOp::SubWithOverflow => hir::BinOpKind::Sub,
            BinOp::Mul | BinOp::MulWithOverflow => hir::BinOpKind::Mul,
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
            // We don't have HIR syntax for these.
            BinOp::Cmp
            | BinOp::AddUnchecked
            | BinOp::SubUnchecked
            | BinOp::MulUnchecked
            | BinOp::ShlUnchecked
            | BinOp::ShrUnchecked
            | BinOp::Offset => {
                unreachable!()
            }
        }
    }

    /// If this is a `FooWithOverflow`, return `Some(Foo)`.
    pub fn overflowing_to_wrapping(self) -> Option<BinOp> {
        Some(match self {
            BinOp::AddWithOverflow => BinOp::Add,
            BinOp::SubWithOverflow => BinOp::Sub,
            BinOp::MulWithOverflow => BinOp::Mul,
            _ => return None,
        })
    }

    /// Returns whether this is a `FooWithOverflow`
    pub fn is_overflowing(self) -> bool {
        self.overflowing_to_wrapping().is_some()
    }

    /// If this is a `Foo`, return `Some(FooWithOverflow)`.
    pub fn wrapping_to_overflowing(self) -> Option<BinOp> {
        Some(match self {
            BinOp::Add => BinOp::AddWithOverflow,
            BinOp::Sub => BinOp::SubWithOverflow,
            BinOp::Mul => BinOp::MulWithOverflow,
            _ => return None,
        })
    }
}
