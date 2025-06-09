//! Functionality for statements, operands, places, and things that appear in them.

use tracing::{debug, instrument};

use super::interpret::GlobalAlloc;
use super::*;
use crate::ty::CoroutineArgsExt;

///////////////////////////////////////////////////////////////////////////
// Statements

/// A statement in a basic block, including information about its source code.
#[derive(Clone, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub struct Statement<'tcx> {
    pub source_info: SourceInfo,
    pub kind: StatementKind<'tcx>,
}

impl Statement<'_> {
    /// Changes a statement to a nop. This is both faster than deleting instructions and avoids
    /// invalidating statement indices in `Location`s.
    pub fn make_nop(&mut self) {
        self.kind = StatementKind::Nop
    }
}

impl<'tcx> StatementKind<'tcx> {
    /// Returns a simple string representation of a `StatementKind` variant, independent of any
    /// values it might hold (e.g. `StatementKind::Assign` always returns `"Assign"`).
    pub const fn name(&self) -> &'static str {
        match self {
            StatementKind::Assign(..) => "Assign",
            StatementKind::FakeRead(..) => "FakeRead",
            StatementKind::SetDiscriminant { .. } => "SetDiscriminant",
            StatementKind::Deinit(..) => "Deinit",
            StatementKind::StorageLive(..) => "StorageLive",
            StatementKind::StorageDead(..) => "StorageDead",
            StatementKind::Retag(..) => "Retag",
            StatementKind::PlaceMention(..) => "PlaceMention",
            StatementKind::AscribeUserType(..) => "AscribeUserType",
            StatementKind::Coverage(..) => "Coverage",
            StatementKind::Intrinsic(..) => "Intrinsic",
            StatementKind::ConstEvalCounter => "ConstEvalCounter",
            StatementKind::Nop => "Nop",
            StatementKind::BackwardIncompatibleDropHint { .. } => "BackwardIncompatibleDropHint",
        }
    }
    pub fn as_assign_mut(&mut self) -> Option<&mut (Place<'tcx>, Rvalue<'tcx>)> {
        match self {
            StatementKind::Assign(x) => Some(x),
            _ => None,
        }
    }

    pub fn as_assign(&self) -> Option<&(Place<'tcx>, Rvalue<'tcx>)> {
        match self {
            StatementKind::Assign(x) => Some(x),
            _ => None,
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Places

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

    /// `place_ty.field_ty(tcx, f)` computes the type of a given field.
    ///
    /// Most clients of `PlaceTy` can instead just extract the relevant type
    /// directly from their `PlaceElem`, but some instances of `ProjectionElem<V, T>`
    /// do not carry a `Ty` for `T`.
    ///
    /// Note that the resulting type has not been normalized.
    #[instrument(level = "debug", skip(tcx), ret)]
    pub fn field_ty(
        tcx: TyCtxt<'tcx>,
        self_ty: Ty<'tcx>,
        variant_idx: Option<VariantIdx>,
        f: FieldIdx,
    ) -> Ty<'tcx> {
        if let Some(variant_index) = variant_idx {
            match *self_ty.kind() {
                ty::Adt(adt_def, args) if adt_def.is_enum() => {
                    adt_def.variant(variant_index).fields[f].ty(tcx, args)
                }
                ty::Coroutine(def_id, args) => {
                    let mut variants = args.as_coroutine().state_tys(def_id, tcx);
                    let Some(mut variant) = variants.nth(variant_index.into()) else {
                        bug!("variant {variant_index:?} of coroutine out of range: {self_ty:?}");
                    };

                    variant.nth(f.index()).unwrap_or_else(|| {
                        bug!("field {f:?} out of range of variant: {self_ty:?} {variant_idx:?}")
                    })
                }
                _ => bug!("can't downcast non-adt non-coroutine type: {self_ty:?}"),
            }
        } else {
            match self_ty.kind() {
                ty::Adt(adt_def, args) if !adt_def.is_enum() => {
                    adt_def.non_enum_variant().fields[f].ty(tcx, args)
                }
                ty::Closure(_, args) => args
                    .as_closure()
                    .upvar_tys()
                    .get(f.index())
                    .copied()
                    .unwrap_or_else(|| bug!("field {f:?} out of range: {self_ty:?}")),
                ty::CoroutineClosure(_, args) => args
                    .as_coroutine_closure()
                    .upvar_tys()
                    .get(f.index())
                    .copied()
                    .unwrap_or_else(|| bug!("field {f:?} out of range: {self_ty:?}")),
                // Only prefix fields (upvars and current state) are
                // accessible without a variant index.
                ty::Coroutine(_, args) => {
                    args.as_coroutine().prefix_tys().get(f.index()).copied().unwrap_or_else(|| {
                        bug!("field {f:?} out of range of prefixes for {self_ty}")
                    })
                }
                ty::Tuple(tys) => tys
                    .get(f.index())
                    .copied()
                    .unwrap_or_else(|| bug!("field {f:?} out of range: {self_ty:?}")),
                _ => bug!("can't project out of {self_ty:?}"),
            }
        }
    }

    pub fn multi_projection_ty(
        self,
        tcx: TyCtxt<'tcx>,
        elems: &[PlaceElem<'tcx>],
    ) -> PlaceTy<'tcx> {
        elems.iter().fold(self, |place_ty, &elem| place_ty.projection_ty(tcx, elem))
    }

    /// Convenience wrapper around `projection_ty_core` for `PlaceElem`,
    /// where we can just use the `Ty` that is already stored inline on
    /// field projection elems.
    pub fn projection_ty(self, tcx: TyCtxt<'tcx>, elem: PlaceElem<'tcx>) -> PlaceTy<'tcx> {
        self.projection_ty_core(tcx, &elem, |ty| ty, |_, _, _, ty| ty, |ty| ty)
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
        mut structurally_normalize: impl FnMut(Ty<'tcx>) -> Ty<'tcx>,
        mut handle_field: impl FnMut(Ty<'tcx>, Option<VariantIdx>, FieldIdx, T) -> Ty<'tcx>,
        mut handle_opaque_cast_and_subtype: impl FnMut(T) -> Ty<'tcx>,
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
                let ty = structurally_normalize(self.ty).builtin_deref(true).unwrap_or_else(|| {
                    bug!("deref projection of non-dereferenceable ty {:?}", self)
                });
                PlaceTy::from_ty(ty)
            }
            ProjectionElem::Index(_) | ProjectionElem::ConstantIndex { .. } => {
                PlaceTy::from_ty(structurally_normalize(self.ty).builtin_index().unwrap())
            }
            ProjectionElem::Subslice { from, to, from_end } => {
                PlaceTy::from_ty(match structurally_normalize(self.ty).kind() {
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
            ProjectionElem::Field(f, fty) => PlaceTy::from_ty(handle_field(
                structurally_normalize(self.ty),
                self.variant_index,
                f,
                fty,
            )),
            ProjectionElem::OpaqueCast(ty) => PlaceTy::from_ty(handle_opaque_cast_and_subtype(ty)),
            ProjectionElem::Subtype(ty) => PlaceTy::from_ty(handle_opaque_cast_and_subtype(ty)),

            // FIXME(unsafe_binders): Rename `handle_opaque_cast_and_subtype` to be more general.
            ProjectionElem::UnwrapUnsafeBinder(ty) => {
                PlaceTy::from_ty(handle_opaque_cast_and_subtype(ty))
            }
        };
        debug!("projection_ty self: {:?} elem: {:?} yields: {:?}", self, elem, answer);
        answer
    }
}

impl<V, T> ProjectionElem<V, T> {
    /// Returns `true` if the target of this projection may refer to a different region of memory
    /// than the base.
    fn is_indirect(&self) -> bool {
        match self {
            Self::Deref => true,

            Self::Field(_, _)
            | Self::Index(_)
            | Self::OpaqueCast(_)
            | Self::Subtype(_)
            | Self::ConstantIndex { .. }
            | Self::Subslice { .. }
            | Self::Downcast(_, _)
            | Self::UnwrapUnsafeBinder(..) => false,
        }
    }

    /// Returns `true` if the target of this projection always refers to the same memory region
    /// whatever the state of the program.
    pub fn is_stable_offset(&self) -> bool {
        match self {
            Self::Deref | Self::Index(_) => false,
            Self::Field(_, _)
            | Self::OpaqueCast(_)
            | Self::Subtype(_)
            | Self::ConstantIndex { .. }
            | Self::Subslice { .. }
            | Self::Downcast(_, _)
            | Self::UnwrapUnsafeBinder(..) => true,
        }
    }

    /// Returns `true` if this is a `Downcast` projection with the given `VariantIdx`.
    pub fn is_downcast_to(&self, v: VariantIdx) -> bool {
        matches!(*self, Self::Downcast(_, x) if x == v)
    }

    /// Returns `true` if this is a `Field` projection with the given index.
    pub fn is_field_to(&self, f: FieldIdx) -> bool {
        matches!(*self, Self::Field(x, _) if x == f)
    }

    /// Returns `true` if this is accepted inside `VarDebugInfoContents::Place`.
    pub fn can_use_in_debuginfo(&self) -> bool {
        match self {
            Self::ConstantIndex { from_end: false, .. }
            | Self::Deref
            | Self::Downcast(_, _)
            | Self::Field(_, _) => true,
            Self::ConstantIndex { from_end: true, .. }
            | Self::Index(_)
            | Self::Subtype(_)
            | Self::OpaqueCast(_)
            | Self::Subslice { .. } => false,

            // FIXME(unsafe_binders): Figure this out.
            Self::UnwrapUnsafeBinder(..) => false,
        }
    }
}

/// Alias for projections as they appear in `UserTypeProjection`, where we
/// need neither the `V` parameter for `Index` nor the `T` for `Field`.
pub type ProjectionKind = ProjectionElem<(), ()>;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaceRef<'tcx> {
    pub local: Local,
    pub projection: &'tcx [PlaceElem<'tcx>],
}

// Once we stop implementing `Ord` for `DefId`,
// this impl will be unnecessary. Until then, we'll
// leave this impl in place to prevent re-adding a
// dependency on the `Ord` impl for `DefId`
impl<'tcx> !PartialOrd for PlaceRef<'tcx> {}

impl<'tcx> Place<'tcx> {
    // FIXME change this to a const fn by also making List::empty a const fn.
    pub fn return_place() -> Place<'tcx> {
        Place { local: RETURN_PLACE, projection: List::empty() }
    }

    /// Returns `true` if this `Place` contains a `Deref` projection.
    ///
    /// If `Place::is_indirect` returns false, the caller knows that the `Place` refers to the
    /// same region of memory as its base.
    pub fn is_indirect(&self) -> bool {
        self.projection.iter().any(|elem| elem.is_indirect())
    }

    /// Returns `true` if this `Place`'s first projection is `Deref`.
    ///
    /// This is useful because for MIR phases `AnalysisPhase::PostCleanup` and later,
    /// `Deref` projections can only occur as the first projection. In that case this method
    /// is equivalent to `is_indirect`, but faster.
    pub fn is_indirect_first_projection(&self) -> bool {
        self.as_ref().is_indirect_first_projection()
    }

    /// Finds the innermost `Local` from this `Place`, *if* it is either a local itself or
    /// a single deref of a local.
    #[inline(always)]
    pub fn local_or_deref_local(&self) -> Option<Local> {
        self.as_ref().local_or_deref_local()
    }

    /// If this place represents a local variable like `_X` with no
    /// projections, return `Some(_X)`.
    #[inline(always)]
    pub fn as_local(&self) -> Option<Local> {
        self.as_ref().as_local()
    }

    #[inline]
    pub fn as_ref(&self) -> PlaceRef<'tcx> {
        PlaceRef { local: self.local, projection: self.projection }
    }

    /// Iterate over the projections in evaluation order, i.e., the first element is the base with
    /// its projection and then subsequently more projections are added.
    /// As a concrete example, given the place a.b.c, this would yield:
    /// - (a, .b)
    /// - (a.b, .c)
    ///
    /// Given a place without projections, the iterator is empty.
    #[inline]
    pub fn iter_projections(
        self,
    ) -> impl Iterator<Item = (PlaceRef<'tcx>, PlaceElem<'tcx>)> + DoubleEndedIterator {
        self.as_ref().iter_projections()
    }

    /// Generates a new place by appending `more_projections` to the existing ones
    /// and interning the result.
    pub fn project_deeper(self, more_projections: &[PlaceElem<'tcx>], tcx: TyCtxt<'tcx>) -> Self {
        if more_projections.is_empty() {
            return self;
        }

        self.as_ref().project_deeper(more_projections, tcx)
    }

    pub fn ty_from<D: ?Sized>(
        local: Local,
        projection: &[PlaceElem<'tcx>],
        local_decls: &D,
        tcx: TyCtxt<'tcx>,
    ) -> PlaceTy<'tcx>
    where
        D: HasLocalDecls<'tcx>,
    {
        PlaceTy::from_ty(local_decls.local_decls()[local].ty).multi_projection_ty(tcx, projection)
    }

    pub fn ty<D: ?Sized>(&self, local_decls: &D, tcx: TyCtxt<'tcx>) -> PlaceTy<'tcx>
    where
        D: HasLocalDecls<'tcx>,
    {
        Place::ty_from(self.local, self.projection, local_decls, tcx)
    }
}

impl From<Local> for Place<'_> {
    #[inline]
    fn from(local: Local) -> Self {
        Place { local, projection: List::empty() }
    }
}

impl<'tcx> PlaceRef<'tcx> {
    /// Finds the innermost `Local` from this `Place`, *if* it is either a local itself or
    /// a single deref of a local.
    pub fn local_or_deref_local(&self) -> Option<Local> {
        match *self {
            PlaceRef { local, projection: [] }
            | PlaceRef { local, projection: [ProjectionElem::Deref] } => Some(local),
            _ => None,
        }
    }

    /// Returns `true` if this `Place` contains a `Deref` projection.
    ///
    /// If `Place::is_indirect` returns false, the caller knows that the `Place` refers to the
    /// same region of memory as its base.
    pub fn is_indirect(&self) -> bool {
        self.projection.iter().any(|elem| elem.is_indirect())
    }

    /// Returns `true` if this `Place`'s first projection is `Deref`.
    ///
    /// This is useful because for MIR phases `AnalysisPhase::PostCleanup` and later,
    /// `Deref` projections can only occur as the first projection. In that case this method
    /// is equivalent to `is_indirect`, but faster.
    pub fn is_indirect_first_projection(&self) -> bool {
        // To make sure this is not accidentally used in wrong mir phase
        debug_assert!(
            self.projection.is_empty() || !self.projection[1..].contains(&PlaceElem::Deref)
        );
        self.projection.first() == Some(&PlaceElem::Deref)
    }

    /// If this place represents a local variable like `_X` with no
    /// projections, return `Some(_X)`.
    #[inline]
    pub fn as_local(&self) -> Option<Local> {
        match *self {
            PlaceRef { local, projection: [] } => Some(local),
            _ => None,
        }
    }

    #[inline]
    pub fn to_place(&self, tcx: TyCtxt<'tcx>) -> Place<'tcx> {
        Place { local: self.local, projection: tcx.mk_place_elems(self.projection) }
    }

    #[inline]
    pub fn last_projection(&self) -> Option<(PlaceRef<'tcx>, PlaceElem<'tcx>)> {
        if let &[ref proj_base @ .., elem] = self.projection {
            Some((PlaceRef { local: self.local, projection: proj_base }, elem))
        } else {
            None
        }
    }

    /// Iterate over the projections in evaluation order, i.e., the first element is the base with
    /// its projection and then subsequently more projections are added.
    /// As a concrete example, given the place a.b.c, this would yield:
    /// - (a, .b)
    /// - (a.b, .c)
    ///
    /// Given a place without projections, the iterator is empty.
    #[inline]
    pub fn iter_projections(
        self,
    ) -> impl Iterator<Item = (PlaceRef<'tcx>, PlaceElem<'tcx>)> + DoubleEndedIterator {
        self.projection.iter().enumerate().map(move |(i, proj)| {
            let base = PlaceRef { local: self.local, projection: &self.projection[..i] };
            (base, *proj)
        })
    }

    /// Generates a new place by appending `more_projections` to the existing ones
    /// and interning the result.
    pub fn project_deeper(
        self,
        more_projections: &[PlaceElem<'tcx>],
        tcx: TyCtxt<'tcx>,
    ) -> Place<'tcx> {
        let mut v: Vec<PlaceElem<'tcx>>;

        let new_projections = if self.projection.is_empty() {
            more_projections
        } else {
            v = Vec::with_capacity(self.projection.len() + more_projections.len());
            v.extend(self.projection);
            v.extend(more_projections);
            &v
        };

        Place { local: self.local, projection: tcx.mk_place_elems(new_projections) }
    }

    pub fn ty<D: ?Sized>(&self, local_decls: &D, tcx: TyCtxt<'tcx>) -> PlaceTy<'tcx>
    where
        D: HasLocalDecls<'tcx>,
    {
        Place::ty_from(self.local, self.projection, local_decls, tcx)
    }
}

impl From<Local> for PlaceRef<'_> {
    #[inline]
    fn from(local: Local) -> Self {
        PlaceRef { local, projection: &[] }
    }
}

///////////////////////////////////////////////////////////////////////////
// Operands

impl<'tcx> Operand<'tcx> {
    /// Convenience helper to make a constant that refers to the fn
    /// with given `DefId` and args. Since this is used to synthesize
    /// MIR, assumes `user_ty` is None.
    pub fn function_handle(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        args: impl IntoIterator<Item = GenericArg<'tcx>>,
        span: Span,
    ) -> Self {
        let ty = Ty::new_fn_def(tcx, def_id, args);
        Operand::Constant(Box::new(ConstOperand {
            span,
            user_ty: None,
            const_: Const::Val(ConstValue::ZeroSized, ty),
        }))
    }

    pub fn is_move(&self) -> bool {
        matches!(self, Operand::Move(..))
    }

    /// Convenience helper to make a literal-like constant from a given scalar value.
    /// Since this is used to synthesize MIR, assumes `user_ty` is None.
    pub fn const_from_scalar(
        tcx: TyCtxt<'tcx>,
        ty: Ty<'tcx>,
        val: Scalar,
        span: Span,
    ) -> Operand<'tcx> {
        debug_assert!({
            let typing_env = ty::TypingEnv::fully_monomorphized();
            let type_size = tcx
                .layout_of(typing_env.as_query_input(ty))
                .unwrap_or_else(|e| panic!("could not compute layout for {ty:?}: {e:?}"))
                .size;
            let scalar_size = match val {
                Scalar::Int(int) => int.size(),
                _ => panic!("Invalid scalar type {val:?}"),
            };
            scalar_size == type_size
        });
        Operand::Constant(Box::new(ConstOperand {
            span,
            user_ty: None,
            const_: Const::Val(ConstValue::Scalar(val), ty),
        }))
    }

    pub fn to_copy(&self) -> Self {
        match *self {
            Operand::Copy(_) | Operand::Constant(_) => self.clone(),
            Operand::Move(place) => Operand::Copy(place),
        }
    }

    /// Returns the `Place` that is the target of this `Operand`, or `None` if this `Operand` is a
    /// constant.
    pub fn place(&self) -> Option<Place<'tcx>> {
        match self {
            Operand::Copy(place) | Operand::Move(place) => Some(*place),
            Operand::Constant(_) => None,
        }
    }

    /// Returns the `ConstOperand` that is the target of this `Operand`, or `None` if this `Operand` is a
    /// place.
    pub fn constant(&self) -> Option<&ConstOperand<'tcx>> {
        match self {
            Operand::Constant(x) => Some(&**x),
            Operand::Copy(_) | Operand::Move(_) => None,
        }
    }

    /// Gets the `ty::FnDef` from an operand if it's a constant function item.
    ///
    /// While this is unlikely in general, it's the normal case of what you'll
    /// find as the `func` in a [`TerminatorKind::Call`].
    pub fn const_fn_def(&self) -> Option<(DefId, GenericArgsRef<'tcx>)> {
        let const_ty = self.constant()?.const_.ty();
        if let ty::FnDef(def_id, args) = *const_ty.kind() { Some((def_id, args)) } else { None }
    }

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

impl<'tcx> ConstOperand<'tcx> {
    pub fn check_static_ptr(&self, tcx: TyCtxt<'_>) -> Option<DefId> {
        match self.const_.try_to_scalar() {
            Some(Scalar::Ptr(ptr, _size)) => match tcx.global_alloc(ptr.provenance.alloc_id()) {
                GlobalAlloc::Static(def_id) => {
                    assert!(!tcx.is_thread_local_static(def_id));
                    Some(def_id)
                }
                _ => None,
            },
            _ => None,
        }
    }

    #[inline]
    pub fn ty(&self) -> Ty<'tcx> {
        self.const_.ty()
    }
}

///////////////////////////////////////////////////////////////////////////
/// Rvalues

pub enum RvalueInitializationState {
    Shallow,
    Deep,
}

impl<'tcx> Rvalue<'tcx> {
    /// Returns true if rvalue can be safely removed when the result is unused.
    #[inline]
    pub fn is_safe_to_remove(&self) -> bool {
        match self {
            // Pointer to int casts may be side-effects due to exposing the provenance.
            // While the model is undecided, we should be conservative. See
            // <https://www.ralfj.de/blog/2022/04/11/provenance-exposed.html>
            Rvalue::Cast(CastKind::PointerExposeProvenance, _, _) => false,

            Rvalue::Use(_)
            | Rvalue::CopyForDeref(_)
            | Rvalue::Repeat(_, _)
            | Rvalue::Ref(_, _, _)
            | Rvalue::ThreadLocalRef(_)
            | Rvalue::RawPtr(_, _)
            | Rvalue::Len(_)
            | Rvalue::Cast(
                CastKind::IntToInt
                | CastKind::FloatToInt
                | CastKind::FloatToFloat
                | CastKind::IntToFloat
                | CastKind::FnPtrToPtr
                | CastKind::PtrToPtr
                | CastKind::PointerCoercion(_, _)
                | CastKind::PointerWithExposedProvenance
                | CastKind::Transmute,
                _,
                _,
            )
            | Rvalue::BinaryOp(_, _)
            | Rvalue::NullaryOp(_, _)
            | Rvalue::UnaryOp(_, _)
            | Rvalue::Discriminant(_)
            | Rvalue::Aggregate(_, _)
            | Rvalue::ShallowInitBox(_, _)
            | Rvalue::WrapUnsafeBinder(_, _) => true,
        }
    }

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
            Rvalue::RawPtr(kind, ref place) => {
                let place_ty = place.ty(local_decls, tcx).ty;
                Ty::new_ptr(tcx, place_ty, kind.to_mutbl_lossy())
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
            Rvalue::NullaryOp(NullOp::ContractChecks, _)
            | Rvalue::NullaryOp(NullOp::UbChecks, _) => tcx.types.bool,
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
            Rvalue::WrapUnsafeBinder(_, ty) => ty,
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

impl BorrowKind {
    pub fn mutability(&self) -> Mutability {
        match *self {
            BorrowKind::Shared | BorrowKind::Fake(_) => Mutability::Not,
            BorrowKind::Mut { .. } => Mutability::Mut,
        }
    }

    /// Returns whether borrows represented by this kind are allowed to be split into separate
    /// Reservation and Activation phases.
    pub fn allows_two_phase_borrow(&self) -> bool {
        match *self {
            BorrowKind::Shared
            | BorrowKind::Fake(_)
            | BorrowKind::Mut { kind: MutBorrowKind::Default | MutBorrowKind::ClosureCapture } => {
                false
            }
            BorrowKind::Mut { kind: MutBorrowKind::TwoPhaseBorrow } => true,
        }
    }

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

impl<'tcx> NullOp<'tcx> {
    pub fn ty(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        match self {
            NullOp::SizeOf | NullOp::AlignOf | NullOp::OffsetOf(_) => tcx.types.usize,
            NullOp::UbChecks | NullOp::ContractChecks => tcx.types.bool,
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
                tcx.ty_ordering_enum(DUMMY_SP)
            }
        }
    }
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

impl From<Mutability> for RawPtrKind {
    fn from(other: Mutability) -> Self {
        match other {
            Mutability::Mut => RawPtrKind::Mut,
            Mutability::Not => RawPtrKind::Const,
        }
    }
}

impl RawPtrKind {
    pub fn is_fake(self) -> bool {
        match self {
            RawPtrKind::Mut | RawPtrKind::Const => false,
            RawPtrKind::FakeForPtrMetadata => true,
        }
    }

    pub fn to_mutbl_lossy(self) -> Mutability {
        match self {
            RawPtrKind::Mut => Mutability::Mut,
            RawPtrKind::Const => Mutability::Not,

            // We have no type corresponding to a fake borrow, so use
            // `*const` as an approximation.
            RawPtrKind::FakeForPtrMetadata => Mutability::Not,
        }
    }

    pub fn ptr_str(self) -> &'static str {
        match self {
            RawPtrKind::Mut => "mut",
            RawPtrKind::Const => "const",
            RawPtrKind::FakeForPtrMetadata => "const (fake)",
        }
    }
}
