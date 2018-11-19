/*!
 * Methods for the various MIR types. These are intended for use after
 * building is complete.
 */

use mir::*;
use ty::subst::{Subst, Substs};
use ty::{self, AdtDef, Ty, TyCtxt};
use ty::layout::VariantIdx;
use hir;
use ty::util::IntTypeExt;

#[derive(Copy, Clone, Debug)]
pub enum PlaceTy<'tcx> {
    /// Normal type.
    Ty { ty: Ty<'tcx> },

    /// Downcast to a particular variant of an enum.
    Downcast { adt_def: &'tcx AdtDef,
               substs: &'tcx Substs<'tcx>,
               variant_index: VariantIdx },
}

static_assert!(PLACE_TY_IS_3_PTRS_LARGE:
    mem::size_of::<PlaceTy<'_>>() <= 24
);

impl From<Ty<'tcx>> for PlaceTy<'tcx> {
    fn from(ty: Ty<'tcx>) -> Self {
        PlaceTy::Ty { ty }
    }
}

impl<'a, 'gcx, 'tcx> PlaceTy<'tcx> {
    pub fn to_ty(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Ty<'tcx> {
        match *self {
            PlaceTy::Ty { ty } =>
                ty,
            PlaceTy::Downcast { adt_def, substs, variant_index: _ } =>
                tcx.mk_adt(adt_def, substs),
        }
    }

    /// `place_ty.field_ty(tcx, f)` computes the type at a given field
    /// of a record or enum-variant. (Most clients of `PlaceTy` can
    /// instead just extract the relevant type directly from their
    /// `PlaceElem`, but some instances of `ProjectionElem<V, T>` do
    /// not carry a `Ty` for `T`.)
    ///
    /// Note that the resulting type has not been normalized.
    pub fn field_ty(self, tcx: TyCtxt<'a, 'gcx, 'tcx>, f: &Field) -> Ty<'tcx>
    {
        // Pass `0` here so it can be used as a "default" variant_index in first arm below
        let answer = match (self, VariantIdx::new(0)) {
            (PlaceTy::Ty {
                ty: &ty::TyS { sty: ty::TyKind::Adt(adt_def, substs), .. } }, variant_index) |
            (PlaceTy::Downcast { adt_def, substs, variant_index }, _) => {
                let variant_def = &adt_def.variants[variant_index];
                let field_def = &variant_def.fields[f.index()];
                field_def.ty(tcx, substs)
            }
            (PlaceTy::Ty { ty }, _) => {
                match ty.sty {
                    ty::Tuple(ref tys) => tys[f.index()],
                    _ => bug!("extracting field of non-tuple non-adt: {:?}", self),
                }
            }
        };
        debug!("field_ty self: {:?} f: {:?} yields: {:?}", self, f, answer);
        answer
    }

    /// Convenience wrapper around `projection_ty_core` for
    /// `PlaceElem`, where we can just use the `Ty` that is already
    /// stored inline on field projection elems.
    pub fn projection_ty(self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                         elem: &PlaceElem<'tcx>)
                         -> PlaceTy<'tcx>
    {
        self.projection_ty_core(tcx, elem, |_, _, ty| ty)
    }

    /// `place_ty.projection_ty_core(tcx, elem, |...| { ... })`
    /// projects `place_ty` onto `elem`, returning the appropriate
    /// `Ty` or downcast variant corresponding to that projection.
    /// The `handle_field` callback must map a `Field` to its `Ty`,
    /// (which should be trivial when `T` = `Ty`).
    pub fn projection_ty_core<V, T>(
        self,
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        elem: &ProjectionElem<'tcx, V, T>,
        mut handle_field: impl FnMut(&Self, &Field, &T) -> Ty<'tcx>)
        -> PlaceTy<'tcx>
    where
        V: ::std::fmt::Debug, T: ::std::fmt::Debug
    {
        let answer = match *elem {
            ProjectionElem::Deref => {
                let ty = self.to_ty(tcx)
                             .builtin_deref(true)
                             .unwrap_or_else(|| {
                                 bug!("deref projection of non-dereferencable ty {:?}", self)
                             })
                             .ty;
                PlaceTy::Ty {
                    ty,
                }
            }
            ProjectionElem::Index(_) | ProjectionElem::ConstantIndex { .. } =>
                PlaceTy::Ty {
                    ty: self.to_ty(tcx).builtin_index().unwrap()
                },
            ProjectionElem::Subslice { from, to } => {
                let ty = self.to_ty(tcx);
                PlaceTy::Ty {
                    ty: match ty.sty {
                        ty::Array(inner, size) => {
                            let size = size.unwrap_usize(tcx);
                            let len = size - (from as u64) - (to as u64);
                            tcx.mk_array(inner, len)
                        }
                        ty::Slice(..) => ty,
                        _ => {
                            bug!("cannot subslice non-array type: `{:?}`", self)
                        }
                    }
                }
            }
            ProjectionElem::Downcast(adt_def1, index) =>
                match self.to_ty(tcx).sty {
                    ty::Adt(adt_def, substs) => {
                        assert!(adt_def.is_enum());
                        assert!(index.as_usize() < adt_def.variants.len());
                        assert_eq!(adt_def, adt_def1);
                        PlaceTy::Downcast { adt_def,
                                            substs,
                                            variant_index: index }
                    }
                    _ => {
                        bug!("cannot downcast non-ADT type: `{:?}`", self)
                    }
                },
            ProjectionElem::Field(ref f, ref fty) =>
                PlaceTy::Ty { ty: handle_field(&self, f, fty) },
        };
        debug!("projection_ty self: {:?} elem: {:?} yields: {:?}", self, elem, answer);
        answer
    }
}

EnumTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for PlaceTy<'tcx> {
        (PlaceTy::Ty) { ty },
        (PlaceTy::Downcast) { adt_def, substs, variant_index },
    }
}

impl<'tcx> Place<'tcx> {
    pub fn ty<'a, 'gcx, D>(&self, local_decls: &D, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> PlaceTy<'tcx>
        where D: HasLocalDecls<'tcx>
    {
        match *self {
            Place::Local(index) =>
                PlaceTy::Ty { ty: local_decls.local_decls()[index].ty },
            Place::Promoted(ref data) => PlaceTy::Ty { ty: data.1 },
            Place::Static(ref data) =>
                PlaceTy::Ty { ty: data.ty },
            Place::Projection(ref proj) =>
                proj.base.ty(local_decls, tcx).projection_ty(tcx, &proj.elem),
        }
    }

    // If this is a field projection, and the field is being projected from a closure type,
    // then returns the index of the field being projected. Note that this closure will always
    // be `self` in the current MIR, because that is the only time we directly access the fields
    // of a closure type.
    //pub fn is_upvar_field_projection<'cx, 'gcx>(&self, mir: &'cx Mir<'tcx>,
    //                                            tcx: &TyCtxt<'cx, 'gcx, 'tcx>) -> Option<Field> {
    //    let (place, by_ref) = if let Place::Projection(ref proj) = self {
    //        if let ProjectionElem::Deref = proj.elem {
    //            (&proj.base, true)
    //        } else {
    //            (self, false)
    //        }
    //    } else {
    //        (self, false)
    //    };

    //    match place {
    //        Place::Projection(ref proj) => match proj.elem {
    //            ProjectionElem::Field(field, _ty) => {
    //                let base_ty = proj.base.ty(mir, *tcx).to_ty(*tcx);

    //                if (base_ty.is_closure() || base_ty.is_generator()) &&
    //                    (!by_ref || mir.upvar_decls[field.index()].by_ref)
    //                {
    //                    Some(field)
    //                } else {
    //                    None
    //                }
    //            },
    //            _ => None,
    //        }
    //        _ => None,
    //    }
    //}
}

impl<'tcx> PlaceBase<'tcx> {
    pub fn ty(&self, local_decls: &impl HasLocalDecls<'tcx>) -> Ty<'tcx> {
        match self {
            PlaceBase::Local(index) => local_decls.local_decls()[*index].ty,
            PlaceBase::Promoted(data) => data.1,
            PlaceBase::Static(data) => data.ty,
        }
    }

    pub fn ty_with_projections(
        &self,
        local_decls: &impl HasLocalDecls<'tcx>,
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        elems: impl Iterator<Item = &'tcx PlaceElem<'tcx>>,
    ) -> Ty<'tcx> {
        let mut base_ty = self.ty(local_decls);

        for elem in elems {
            base_ty = PlaceTy::from(base_ty).projection_ty(tcx, elem).to_ty(tcx);
        }

        base_ty
    }
}

impl<'tcx> NeoPlace<'tcx> {
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
        // Look for either *(Place.field) or Place.field,
        // where P is a place with closure type,
        // these sorts of places represent accesses to the closure's captured upvars.

        // unwrap inner place when Deref matched.
        // *(closure.field)
        // ^         ^^^^^ inner projection_elem
        // |-- Deref
        let (elems, by_ref) =
            if let Some(ProjectionElem::Deref) = self.elems.last() {
                (&self.elems[..self.elems.len()-1], true)
        } else {
            (&self.elems[..], false)
        };
        let mut elems = elems.iter().rev();

        // closure.field
        //         ^^^^^
        if let Some(ProjectionElem::Field(field, _ty)) = elems.next() {
            let base_ty = self.base.ty_with_projections(mir, *tcx, elems.rev());
            if (base_ty.is_closure() || base_ty.is_generator()) &&
                (!by_ref || mir.upvar_decls[field.index()].by_ref)
            {
                Some(*field)
            } else {
                None
            }
        } else {
            None
        }
    }

    // Return the base of place without projections
    pub fn as_place_base(&self) -> Option<&PlaceBase<'tcx>>{
        if self.elems.is_empty() {
            Some(&self.base)
        } else {
            None
        }
    }

    // for a place_elem returns it's prefix path
    // Base.[a, b, c]
    //          ^-- place_elem
    // ^^^^^^^-- prefix
    pub fn prefix<'cx, 'gcx>(
        &self,
        tcx: TyCtxt<'cx, 'gcx, 'tcx>,
        elem_index: usize,
    ) -> Self {
        if elem_index < 1 {
            // Base.[a]
            //       ^-- elems[0]
            Self {
                base: self.clone().base,
                elems: List::empty(),
            }
        } else {
            Self {
                base: self.clone().base,
                elems: tcx.mk_place_elems(self.elems[..elem_index].iter()),
            }
        }
    }

    pub fn prefixes<'cx, 'gcx>(
        &self,
        tcx: TyCtxt<'cx, 'gcx, 'tcx>,
    ) -> Vec<Self> {
        // only works for place with projections
        assert!(!self.elems.is_empty());

        self.elems
            .iter()
            .enumerate()
            .map(|(elem_index, _)| self.prefix(tcx, elem_index))
            .collect::<Vec<NeoPlace<'tcx>>>()
    }

    // Generate types of sub-places
    //
    // Base.[a, b]
    // ^^^^-- base_ty
    // ^^^^^^^-- elem1_ty
    // ^^^^^^^^^^--elem2_ty
    //
    // [base_ty, elem1_ty, elem2_ty]
    pub fn place_tys<'cx, 'gcx>(
        &self,
        mir: &'cx Mir<'tcx>,
        tcx: TyCtxt<'cx, 'gcx, 'tcx>,
    ) -> Vec<PlaceTy<'tcx>> {
        let mut base_ty = PlaceTy::from(self.base.ty(mir));
        let mut place_tys: Vec<PlaceTy<'tcx>> = vec![base_ty];
        for elem in self.elems.iter() {
            let elem_ty = base_ty.projection_ty(tcx, elem);
            place_tys.push(elem_ty);
            base_ty = elem_ty;
        }
        place_tys
    }
}

pub enum RvalueInitializationState {
    Shallow,
    Deep
}

impl<'tcx> Rvalue<'tcx> {
    pub fn ty<'a, 'gcx, D>(&self, local_decls: &D, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Ty<'tcx>
        where D: HasLocalDecls<'tcx>
    {
        match *self {
            Rvalue::Use(ref operand) => operand.ty(local_decls, tcx),
            Rvalue::Repeat(ref operand, count) => {
                tcx.mk_array(operand.ty(local_decls, tcx), count)
            }
            Rvalue::Ref(reg, bk, ref place) => {
                let place_ty = place.ty(local_decls, tcx).to_ty(tcx);
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
                let ty = place.ty(local_decls, tcx).to_ty(tcx);
                if let ty::Adt(adt_def, _) = ty.sty {
                    adt_def.repr.discr_type().to_ty(tcx)
                } else {
                    // This can only be `0`, for now, so `u8` will suffice.
                    tcx.types.u8
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
    /// Returns whether this rvalue is deeply initialized (most rvalues) or
    /// whether its only shallowly initialized (`Rvalue::Box`).
    pub fn initialization_state(&self) -> RvalueInitializationState {
        match *self {
            Rvalue::NullaryOp(NullOp::Box, _) => RvalueInitializationState::Shallow,
            _ => RvalueInitializationState::Deep
        }
    }
}

impl<'tcx> Operand<'tcx> {
    pub fn ty<'a, 'gcx, D>(&self, local_decls: &D, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Ty<'tcx>
        where D: HasLocalDecls<'tcx>
    {
        match self {
            &Operand::Copy(ref l) |
            &Operand::Move(ref l) => l.ty(local_decls, tcx).to_ty(tcx),
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
