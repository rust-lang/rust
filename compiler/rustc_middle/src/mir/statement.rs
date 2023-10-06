/// Functionality for statements, operands, places, and things that appear in them.
use super::{interpret::GlobalAlloc, *};

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

    /// Changes a statement to a nop and returns the original statement.
    #[must_use = "If you don't need the statement, use `make_nop` instead"]
    pub fn replace_nop(&mut self) -> Self {
        Statement {
            source_info: self.source_info,
            kind: mem::replace(&mut self.kind, StatementKind::Nop),
        }
    }
}

impl<'tcx> StatementKind<'tcx> {
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
            | Self::Downcast(_, _) => false,
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
            | Self::Downcast(_, _) => true,
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
        PlaceRef { local: self.local, projection: &self.projection }
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
            let param_env_and_ty = ty::ParamEnv::empty().and(ty);
            let type_size = tcx
                .layout_of(param_env_and_ty)
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
}

impl<'tcx> ConstOperand<'tcx> {
    pub fn check_static_ptr(&self, tcx: TyCtxt<'_>) -> Option<DefId> {
        match self.const_.try_to_scalar() {
            Some(Scalar::Ptr(ptr, _size)) => match tcx.global_alloc(ptr.provenance) {
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

impl<'tcx> Rvalue<'tcx> {
    /// Returns true if rvalue can be safely removed when the result is unused.
    #[inline]
    pub fn is_safe_to_remove(&self) -> bool {
        match self {
            // Pointer to int casts may be side-effects due to exposing the provenance.
            // While the model is undecided, we should be conservative. See
            // <https://www.ralfj.de/blog/2022/04/11/provenance-exposed.html>
            Rvalue::Cast(CastKind::PointerExposeAddress, _, _) => false,

            Rvalue::Use(_)
            | Rvalue::CopyForDeref(_)
            | Rvalue::Repeat(_, _)
            | Rvalue::Ref(_, _, _)
            | Rvalue::ThreadLocalRef(_)
            | Rvalue::AddressOf(_, _)
            | Rvalue::Len(_)
            | Rvalue::Cast(
                CastKind::IntToInt
                | CastKind::FloatToInt
                | CastKind::FloatToFloat
                | CastKind::IntToFloat
                | CastKind::FnPtrToPtr
                | CastKind::PtrToPtr
                | CastKind::PointerCoercion(_)
                | CastKind::PointerFromExposedAddress
                | CastKind::DynStar
                | CastKind::Transmute,
                _,
                _,
            )
            | Rvalue::BinaryOp(_, _)
            | Rvalue::CheckedBinaryOp(_, _)
            | Rvalue::NullaryOp(_, _)
            | Rvalue::UnaryOp(_, _)
            | Rvalue::Discriminant(_)
            | Rvalue::Aggregate(_, _)
            | Rvalue::ShallowInitBox(_, _) => true,
        }
    }
}

impl BorrowKind {
    pub fn mutability(&self) -> Mutability {
        match *self {
            BorrowKind::Shared | BorrowKind::Shallow => Mutability::Not,
            BorrowKind::Mut { .. } => Mutability::Mut,
        }
    }

    pub fn allows_two_phase_borrow(&self) -> bool {
        match *self {
            BorrowKind::Shared
            | BorrowKind::Shallow
            | BorrowKind::Mut { kind: MutBorrowKind::Default | MutBorrowKind::ClosureCapture } => {
                false
            }
            BorrowKind::Mut { kind: MutBorrowKind::TwoPhaseBorrow } => true,
        }
    }
}
