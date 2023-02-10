//! This module contains implements of the `Lift` and `TypeFoldable`
//! traits for various types in the Rust compiler. Most are written by
//! hand, though we've recently added some macros and proc-macros to help with the tedium.

use crate::mir::interpret;
use crate::mir::{Field, ProjectionKind};
use crate::ty::fold::{FallibleTypeFolder, TypeFoldable, TypeSuperFoldable};
use crate::ty::print::{with_no_trimmed_paths, FmtPrinter, Printer};
use crate::ty::visit::{TypeSuperVisitable, TypeVisitable, TypeVisitor};
use crate::ty::{self, AliasTy, InferConst, Lift, Term, TermKind, Ty, TyCtxt};
use rustc_data_structures::functor::IdFunctor;
use rustc_hir::def::Namespace;
use rustc_index::vec::{Idx, IndexVec};
use rustc_target::abi::TyAndLayout;

use std::fmt;
use std::mem::ManuallyDrop;
use std::ops::ControlFlow;
use std::rc::Rc;
use std::sync::Arc;

impl fmt::Debug for ty::TraitDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ty::tls::with(|tcx| {
            with_no_trimmed_paths!({
                f.write_str(
                    &FmtPrinter::new(tcx, Namespace::TypeNS)
                        .print_def_path(self.def_id, &[])?
                        .into_buffer(),
                )
            })
        })
    }
}

impl<'tcx> fmt::Debug for ty::AdtDef<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ty::tls::with(|tcx| {
            with_no_trimmed_paths!({
                f.write_str(
                    &FmtPrinter::new(tcx, Namespace::TypeNS)
                        .print_def_path(self.did(), &[])?
                        .into_buffer(),
                )
            })
        })
    }
}

impl fmt::Debug for ty::UpvarId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = ty::tls::with(|tcx| tcx.hir().name(self.var_path.hir_id));
        write!(f, "UpvarId({:?};`{}`;{:?})", self.var_path.hir_id, name, self.closure_expr_id)
    }
}

impl<'tcx> fmt::Debug for ty::ExistentialTraitRef<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        with_no_trimmed_paths!(fmt::Display::fmt(self, f))
    }
}

impl<'tcx> fmt::Debug for ty::adjustment::Adjustment<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} -> {}", self.kind, self.target)
    }
}

impl fmt::Debug for ty::BoundRegionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ty::BrAnon(n, span) => write!(f, "BrAnon({n:?}, {span:?})"),
            ty::BrNamed(did, name) => {
                if did.is_crate_root() {
                    write!(f, "BrNamed({})", name)
                } else {
                    write!(f, "BrNamed({:?}, {})", did, name)
                }
            }
            ty::BrEnv => write!(f, "BrEnv"),
        }
    }
}

impl fmt::Debug for ty::FreeRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReFree({:?}, {:?})", self.scope, self.bound_region)
    }
}

impl<'tcx> fmt::Debug for ty::FnSig<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:?}; c_variadic: {})->{:?}", self.inputs(), self.c_variadic, self.output())
    }
}

impl<'tcx> fmt::Debug for ty::ConstVid<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_#{}c", self.index)
    }
}

impl<'tcx> fmt::Debug for ty::TraitRef<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        with_no_trimmed_paths!(fmt::Display::fmt(self, f))
    }
}

impl<'tcx> fmt::Debug for Ty<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        with_no_trimmed_paths!(fmt::Display::fmt(self, f))
    }
}

impl fmt::Debug for ty::ParamTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/#{}", self.name, self.index)
    }
}

impl fmt::Debug for ty::ParamConst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/#{}", self.name, self.index)
    }
}

impl<'tcx> fmt::Debug for ty::TraitPredicate<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let ty::BoundConstness::ConstIfConst = self.constness {
            write!(f, "~const ")?;
        }
        write!(f, "TraitPredicate({:?}, polarity:{:?})", self.trait_ref, self.polarity)
    }
}

impl<'tcx> fmt::Debug for ty::ProjectionPredicate<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ProjectionPredicate({:?}, {:?})", self.projection_ty, self.term)
    }
}

impl<'tcx> fmt::Debug for ty::Predicate<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.kind())
    }
}

impl<'tcx> fmt::Debug for ty::Clause<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ty::Clause::Trait(ref a) => a.fmt(f),
            ty::Clause::RegionOutlives(ref pair) => pair.fmt(f),
            ty::Clause::TypeOutlives(ref pair) => pair.fmt(f),
            ty::Clause::Projection(ref pair) => pair.fmt(f),
        }
    }
}

impl<'tcx> fmt::Debug for ty::PredicateKind<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ty::PredicateKind::Clause(ref a) => a.fmt(f),
            ty::PredicateKind::Subtype(ref pair) => pair.fmt(f),
            ty::PredicateKind::Coerce(ref pair) => pair.fmt(f),
            ty::PredicateKind::WellFormed(data) => write!(f, "WellFormed({:?})", data),
            ty::PredicateKind::ObjectSafe(trait_def_id) => {
                write!(f, "ObjectSafe({:?})", trait_def_id)
            }
            ty::PredicateKind::ClosureKind(closure_def_id, closure_substs, kind) => {
                write!(f, "ClosureKind({:?}, {:?}, {:?})", closure_def_id, closure_substs, kind)
            }
            ty::PredicateKind::ConstEvaluatable(ct) => {
                write!(f, "ConstEvaluatable({ct:?})")
            }
            ty::PredicateKind::ConstEquate(c1, c2) => write!(f, "ConstEquate({:?}, {:?})", c1, c2),
            ty::PredicateKind::TypeWellFormedFromEnv(ty) => {
                write!(f, "TypeWellFormedFromEnv({:?})", ty)
            }
            ty::PredicateKind::Ambiguous => write!(f, "Ambiguous"),
            ty::PredicateKind::AliasEq(t1, t2) => write!(f, "AliasEq({t1:?}, {t2:?})"),
        }
    }
}

impl<'tcx> fmt::Debug for AliasTy<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AliasTy")
            .field("substs", &self.substs)
            .field("def_id", &self.def_id)
            .finish()
    }
}

///////////////////////////////////////////////////////////////////////////
// Atomic structs
//
// For things that don't carry any arena-allocated data (and are
// copy...), just add them to this list.

TrivialTypeTraversalAndLiftImpls! {
    (),
    bool,
    usize,
    ::rustc_target::abi::VariantIdx,
    u16,
    u32,
    u64,
    String,
    crate::middle::region::Scope,
    crate::ty::FloatTy,
    ::rustc_ast::InlineAsmOptions,
    ::rustc_ast::InlineAsmTemplatePiece,
    ::rustc_ast::NodeId,
    ::rustc_span::symbol::Symbol,
    ::rustc_hir::def::Res,
    ::rustc_hir::def_id::DefId,
    ::rustc_hir::def_id::LocalDefId,
    ::rustc_hir::HirId,
    ::rustc_hir::MatchSource,
    ::rustc_hir::Mutability,
    ::rustc_hir::Unsafety,
    ::rustc_target::asm::InlineAsmRegOrRegClass,
    ::rustc_target::spec::abi::Abi,
    crate::mir::coverage::ExpressionOperandId,
    crate::mir::coverage::CounterValueReference,
    crate::mir::coverage::InjectedExpressionId,
    crate::mir::coverage::InjectedExpressionIndex,
    crate::mir::coverage::MappedExpressionIndex,
    crate::mir::Local,
    crate::mir::Promoted,
    crate::traits::Reveal,
    crate::ty::adjustment::AutoBorrowMutability,
    crate::ty::AdtKind,
    crate::ty::BoundConstness,
    // Including `BoundRegionKind` is a *bit* dubious, but direct
    // references to bound region appear in `ty::Error`, and aren't
    // really meant to be folded. In general, we can only fold a fully
    // general `Region`.
    crate::ty::BoundRegionKind,
    crate::ty::AssocItem,
    crate::ty::AssocKind,
    crate::ty::AliasKind,
    crate::ty::Placeholder<crate::ty::BoundRegionKind>,
    crate::ty::Placeholder<crate::ty::BoundTyKind>,
    crate::ty::ClosureKind,
    crate::ty::FreeRegion,
    crate::ty::InferTy,
    crate::ty::IntVarValue,
    crate::ty::ParamConst,
    crate::ty::ParamTy,
    crate::ty::adjustment::PointerCast,
    crate::ty::RegionVid,
    crate::ty::UniverseIndex,
    crate::ty::Variance,
    ::rustc_span::Span,
    ::rustc_errors::ErrorGuaranteed,
    Field,
    interpret::Scalar,
    rustc_target::abi::Size,
    rustc_type_ir::DebruijnIndex,
    ty::BoundVar,
    ty::Placeholder<ty::BoundVar>,
}

TrivialTypeTraversalAndLiftImpls! {
    for<'tcx> {
        ty::ValTree<'tcx>,
    }
}

///////////////////////////////////////////////////////////////////////////
// Lift implementations

impl<'tcx, A: Lift<'tcx>, B: Lift<'tcx>> Lift<'tcx> for (A, B) {
    type Lifted = (A::Lifted, B::Lifted);
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        Some((tcx.lift(self.0)?, tcx.lift(self.1)?))
    }
}

impl<'tcx, A: Lift<'tcx>, B: Lift<'tcx>, C: Lift<'tcx>> Lift<'tcx> for (A, B, C) {
    type Lifted = (A::Lifted, B::Lifted, C::Lifted);
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        Some((tcx.lift(self.0)?, tcx.lift(self.1)?, tcx.lift(self.2)?))
    }
}

impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for Option<T> {
    type Lifted = Option<T::Lifted>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        Some(match self {
            Some(x) => Some(tcx.lift(x)?),
            None => None,
        })
    }
}

impl<'tcx, T: Lift<'tcx>, E: Lift<'tcx>> Lift<'tcx> for Result<T, E> {
    type Lifted = Result<T::Lifted, E::Lifted>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        match self {
            Ok(x) => tcx.lift(x).map(Ok),
            Err(e) => tcx.lift(e).map(Err),
        }
    }
}

impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for Box<T> {
    type Lifted = Box<T::Lifted>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        Some(Box::new(tcx.lift(*self)?))
    }
}

impl<'tcx, T: Lift<'tcx> + Clone> Lift<'tcx> for Rc<T> {
    type Lifted = Rc<T::Lifted>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        Some(Rc::new(tcx.lift(self.as_ref().clone())?))
    }
}

impl<'tcx, T: Lift<'tcx> + Clone> Lift<'tcx> for Arc<T> {
    type Lifted = Arc<T::Lifted>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        Some(Arc::new(tcx.lift(self.as_ref().clone())?))
    }
}
impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for Vec<T> {
    type Lifted = Vec<T::Lifted>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        self.into_iter().map(|v| tcx.lift(v)).collect()
    }
}

impl<'tcx, I: Idx, T: Lift<'tcx>> Lift<'tcx> for IndexVec<I, T> {
    type Lifted = IndexVec<I, T::Lifted>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        self.into_iter().map(|e| tcx.lift(e)).collect()
    }
}

impl<'a, 'tcx> Lift<'tcx> for Term<'a> {
    type Lifted = ty::Term<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        Some(
            match self.unpack() {
                TermKind::Ty(ty) => TermKind::Ty(tcx.lift(ty)?),
                TermKind::Const(c) => TermKind::Const(tcx.lift(c)?),
            }
            .pack(),
        )
    }
}
impl<'a, 'tcx> Lift<'tcx> for ty::ParamEnv<'a> {
    type Lifted = ty::ParamEnv<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(self.caller_bounds())
            .map(|caller_bounds| ty::ParamEnv::new(caller_bounds, self.reveal(), self.constness()))
    }
}

///////////////////////////////////////////////////////////////////////////
// TypeFoldable implementations.

/// AdtDefs are basically the same as a DefId.
impl<'tcx> TypeFoldable<'tcx> for ty::AdtDef<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, _folder: &mut F) -> Result<Self, F::Error> {
        Ok(self)
    }
}

impl<'tcx> TypeVisitable<'tcx> for ty::AdtDef<'tcx> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, _visitor: &mut V) -> ControlFlow<V::BreakTy> {
        ControlFlow::Continue(())
    }
}

impl<'tcx, T: TypeFoldable<'tcx>, U: TypeFoldable<'tcx>> TypeFoldable<'tcx> for (T, U) {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(
        self,
        folder: &mut F,
    ) -> Result<(T, U), F::Error> {
        Ok((self.0.try_fold_with(folder)?, self.1.try_fold_with(folder)?))
    }
}

impl<'tcx, T: TypeVisitable<'tcx>, U: TypeVisitable<'tcx>> TypeVisitable<'tcx> for (T, U) {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.0.visit_with(visitor)?;
        self.1.visit_with(visitor)
    }
}

impl<'tcx, A: TypeFoldable<'tcx>, B: TypeFoldable<'tcx>, C: TypeFoldable<'tcx>> TypeFoldable<'tcx>
    for (A, B, C)
{
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(
        self,
        folder: &mut F,
    ) -> Result<(A, B, C), F::Error> {
        Ok((
            self.0.try_fold_with(folder)?,
            self.1.try_fold_with(folder)?,
            self.2.try_fold_with(folder)?,
        ))
    }
}

impl<'tcx, A: TypeVisitable<'tcx>, B: TypeVisitable<'tcx>, C: TypeVisitable<'tcx>>
    TypeVisitable<'tcx> for (A, B, C)
{
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.0.visit_with(visitor)?;
        self.1.visit_with(visitor)?;
        self.2.visit_with(visitor)
    }
}

EnumTypeTraversalImpl! {
    impl<'tcx, T> TypeFoldable<'tcx> for Option<T> {
        (Some)(a),
        (None),
    } where T: TypeFoldable<'tcx>
}
EnumTypeTraversalImpl! {
    impl<'tcx, T> TypeVisitable<'tcx> for Option<T> {
        (Some)(a),
        (None),
    } where T: TypeVisitable<'tcx>
}

EnumTypeTraversalImpl! {
    impl<'tcx, T, E> TypeFoldable<'tcx> for Result<T, E> {
        (Ok)(a),
        (Err)(a),
    } where T: TypeFoldable<'tcx>, E: TypeFoldable<'tcx>,
}
EnumTypeTraversalImpl! {
    impl<'tcx, T, E> TypeVisitable<'tcx> for Result<T, E> {
        (Ok)(a),
        (Err)(a),
    } where T: TypeVisitable<'tcx>, E: TypeVisitable<'tcx>,
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Rc<T> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(
        mut self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        // We merely want to replace the contained `T`, if at all possible,
        // so that we don't needlessly allocate a new `Rc` or indeed clone
        // the contained type.
        unsafe {
            // First step is to ensure that we have a unique reference to
            // the contained type, which `Rc::make_mut` will accomplish (by
            // allocating a new `Rc` and cloning the `T` only if required).
            // This is done *before* casting to `Rc<ManuallyDrop<T>>` so that
            // panicking during `make_mut` does not leak the `T`.
            Rc::make_mut(&mut self);

            // Casting to `Rc<ManuallyDrop<T>>` is safe because `ManuallyDrop`
            // is `repr(transparent)`.
            let ptr = Rc::into_raw(self).cast::<ManuallyDrop<T>>();
            let mut unique = Rc::from_raw(ptr);

            // Call to `Rc::make_mut` above guarantees that `unique` is the
            // sole reference to the contained value, so we can avoid doing
            // a checked `get_mut` here.
            let slot = Rc::get_mut_unchecked(&mut unique);

            // Semantically move the contained type out from `unique`, fold
            // it, then move the folded value back into `unique`. Should
            // folding fail, `ManuallyDrop` ensures that the "moved-out"
            // value is not re-dropped.
            let owned = ManuallyDrop::take(slot);
            let folded = owned.try_fold_with(folder)?;
            *slot = ManuallyDrop::new(folded);

            // Cast back to `Rc<T>`.
            Ok(Rc::from_raw(Rc::into_raw(unique).cast()))
        }
    }
}

impl<'tcx, T: TypeVisitable<'tcx>> TypeVisitable<'tcx> for Rc<T> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        (**self).visit_with(visitor)
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Arc<T> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(
        mut self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        // We merely want to replace the contained `T`, if at all possible,
        // so that we don't needlessly allocate a new `Arc` or indeed clone
        // the contained type.
        unsafe {
            // First step is to ensure that we have a unique reference to
            // the contained type, which `Arc::make_mut` will accomplish (by
            // allocating a new `Arc` and cloning the `T` only if required).
            // This is done *before* casting to `Arc<ManuallyDrop<T>>` so that
            // panicking during `make_mut` does not leak the `T`.
            Arc::make_mut(&mut self);

            // Casting to `Arc<ManuallyDrop<T>>` is safe because `ManuallyDrop`
            // is `repr(transparent)`.
            let ptr = Arc::into_raw(self).cast::<ManuallyDrop<T>>();
            let mut unique = Arc::from_raw(ptr);

            // Call to `Arc::make_mut` above guarantees that `unique` is the
            // sole reference to the contained value, so we can avoid doing
            // a checked `get_mut` here.
            let slot = Arc::get_mut_unchecked(&mut unique);

            // Semantically move the contained type out from `unique`, fold
            // it, then move the folded value back into `unique`. Should
            // folding fail, `ManuallyDrop` ensures that the "moved-out"
            // value is not re-dropped.
            let owned = ManuallyDrop::take(slot);
            let folded = owned.try_fold_with(folder)?;
            *slot = ManuallyDrop::new(folded);

            // Cast back to `Arc<T>`.
            Ok(Arc::from_raw(Arc::into_raw(unique).cast()))
        }
    }
}

impl<'tcx, T: TypeVisitable<'tcx>> TypeVisitable<'tcx> for Arc<T> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        (**self).visit_with(visitor)
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Box<T> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, folder: &mut F) -> Result<Self, F::Error> {
        self.try_map_id(|value| value.try_fold_with(folder))
    }
}

impl<'tcx, T: TypeVisitable<'tcx>> TypeVisitable<'tcx> for Box<T> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        (**self).visit_with(visitor)
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Vec<T> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, folder: &mut F) -> Result<Self, F::Error> {
        self.try_map_id(|t| t.try_fold_with(folder))
    }
}

impl<'tcx, T: TypeVisitable<'tcx>> TypeVisitable<'tcx> for Vec<T> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}

impl<'tcx, T: TypeVisitable<'tcx>> TypeVisitable<'tcx> for &[T] {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Box<[T]> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, folder: &mut F) -> Result<Self, F::Error> {
        self.try_map_id(|t| t.try_fold_with(folder))
    }
}

impl<'tcx, T: TypeVisitable<'tcx>> TypeVisitable<'tcx> for Box<[T]> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for ty::Binder<'tcx, T> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, folder: &mut F) -> Result<Self, F::Error> {
        folder.try_fold_binder(self)
    }
}

impl<'tcx, T: TypeVisitable<'tcx>> TypeVisitable<'tcx> for ty::Binder<'tcx, T> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        visitor.visit_binder(self)
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeSuperFoldable<'tcx> for ty::Binder<'tcx, T> {
    fn try_super_fold_with<F: FallibleTypeFolder<'tcx>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        self.try_map_bound(|ty| ty.try_fold_with(folder))
    }
}

impl<'tcx, T: TypeVisitable<'tcx>> TypeSuperVisitable<'tcx> for ty::Binder<'tcx, T> {
    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.as_ref().skip_binder().visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, folder: &mut F) -> Result<Self, F::Error> {
        ty::util::fold_list(self, folder, |tcx, v| tcx.intern_poly_existential_predicates(v))
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::List<ty::Const<'tcx>> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, folder: &mut F) -> Result<Self, F::Error> {
        ty::util::fold_list(self, folder, |tcx, v| tcx.mk_const_list(v.iter()))
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::List<ProjectionKind> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, folder: &mut F) -> Result<Self, F::Error> {
        ty::util::fold_list(self, folder, |tcx, v| tcx.intern_projs(v))
    }
}

impl<'tcx> TypeFoldable<'tcx> for Ty<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, folder: &mut F) -> Result<Self, F::Error> {
        folder.try_fold_ty(self)
    }
}

impl<'tcx> TypeVisitable<'tcx> for Ty<'tcx> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        visitor.visit_ty(*self)
    }
}

impl<'tcx> TypeSuperFoldable<'tcx> for Ty<'tcx> {
    fn try_super_fold_with<F: FallibleTypeFolder<'tcx>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        let kind = match *self.kind() {
            ty::RawPtr(tm) => ty::RawPtr(tm.try_fold_with(folder)?),
            ty::Array(typ, sz) => ty::Array(typ.try_fold_with(folder)?, sz.try_fold_with(folder)?),
            ty::Slice(typ) => ty::Slice(typ.try_fold_with(folder)?),
            ty::Adt(tid, substs) => ty::Adt(tid, substs.try_fold_with(folder)?),
            ty::Dynamic(trait_ty, region, representation) => ty::Dynamic(
                trait_ty.try_fold_with(folder)?,
                region.try_fold_with(folder)?,
                representation,
            ),
            ty::Tuple(ts) => ty::Tuple(ts.try_fold_with(folder)?),
            ty::FnDef(def_id, substs) => ty::FnDef(def_id, substs.try_fold_with(folder)?),
            ty::FnPtr(f) => ty::FnPtr(f.try_fold_with(folder)?),
            ty::Ref(r, ty, mutbl) => {
                ty::Ref(r.try_fold_with(folder)?, ty.try_fold_with(folder)?, mutbl)
            }
            ty::Generator(did, substs, movability) => {
                ty::Generator(did, substs.try_fold_with(folder)?, movability)
            }
            ty::GeneratorWitness(types) => ty::GeneratorWitness(types.try_fold_with(folder)?),
            ty::GeneratorWitnessMIR(did, substs) => {
                ty::GeneratorWitnessMIR(did, substs.try_fold_with(folder)?)
            }
            ty::Closure(did, substs) => ty::Closure(did, substs.try_fold_with(folder)?),
            ty::Alias(kind, data) => ty::Alias(kind, data.try_fold_with(folder)?),

            ty::Bool
            | ty::Char
            | ty::Str
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Error(_)
            | ty::Infer(_)
            | ty::Param(..)
            | ty::Bound(..)
            | ty::Placeholder(..)
            | ty::Never
            | ty::Foreign(..) => return Ok(self),
        };

        Ok(if *self.kind() == kind { self } else { folder.tcx().mk_ty(kind) })
    }
}

impl<'tcx> TypeSuperVisitable<'tcx> for Ty<'tcx> {
    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        match self.kind() {
            ty::RawPtr(ref tm) => tm.visit_with(visitor),
            ty::Array(typ, sz) => {
                typ.visit_with(visitor)?;
                sz.visit_with(visitor)
            }
            ty::Slice(typ) => typ.visit_with(visitor),
            ty::Adt(_, substs) => substs.visit_with(visitor),
            ty::Dynamic(ref trait_ty, ref reg, _) => {
                trait_ty.visit_with(visitor)?;
                reg.visit_with(visitor)
            }
            ty::Tuple(ts) => ts.visit_with(visitor),
            ty::FnDef(_, substs) => substs.visit_with(visitor),
            ty::FnPtr(ref f) => f.visit_with(visitor),
            ty::Ref(r, ty, _) => {
                r.visit_with(visitor)?;
                ty.visit_with(visitor)
            }
            ty::Generator(_did, ref substs, _) => substs.visit_with(visitor),
            ty::GeneratorWitness(ref types) => types.visit_with(visitor),
            ty::GeneratorWitnessMIR(_did, ref substs) => substs.visit_with(visitor),
            ty::Closure(_did, ref substs) => substs.visit_with(visitor),
            ty::Alias(_, ref data) => data.visit_with(visitor),

            ty::Bool
            | ty::Char
            | ty::Str
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Error(_)
            | ty::Infer(_)
            | ty::Bound(..)
            | ty::Placeholder(..)
            | ty::Param(..)
            | ty::Never
            | ty::Foreign(..) => ControlFlow::Continue(()),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::Region<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, folder: &mut F) -> Result<Self, F::Error> {
        folder.try_fold_region(self)
    }
}

impl<'tcx> TypeVisitable<'tcx> for ty::Region<'tcx> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        visitor.visit_region(*self)
    }
}

impl<'tcx> TypeSuperFoldable<'tcx> for ty::Region<'tcx> {
    fn try_super_fold_with<F: FallibleTypeFolder<'tcx>>(
        self,
        _folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(self)
    }
}

impl<'tcx> TypeSuperVisitable<'tcx> for ty::Region<'tcx> {
    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, _visitor: &mut V) -> ControlFlow<V::BreakTy> {
        ControlFlow::Continue(())
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::Predicate<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, folder: &mut F) -> Result<Self, F::Error> {
        folder.try_fold_predicate(self)
    }
}

impl<'tcx> TypeVisitable<'tcx> for ty::Predicate<'tcx> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        visitor.visit_predicate(*self)
    }

    #[inline]
    fn has_vars_bound_at_or_above(&self, binder: ty::DebruijnIndex) -> bool {
        self.outer_exclusive_binder() > binder
    }

    #[inline]
    fn has_type_flags(&self, flags: ty::TypeFlags) -> bool {
        self.flags().intersects(flags)
    }
}

impl<'tcx> TypeSuperFoldable<'tcx> for ty::Predicate<'tcx> {
    fn try_super_fold_with<F: FallibleTypeFolder<'tcx>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        let new = self.kind().try_fold_with(folder)?;
        Ok(folder.tcx().reuse_or_mk_predicate(self, new))
    }
}

impl<'tcx> TypeSuperVisitable<'tcx> for ty::Predicate<'tcx> {
    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.kind().visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::List<ty::Predicate<'tcx>> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, folder: &mut F) -> Result<Self, F::Error> {
        ty::util::fold_list(self, folder, |tcx, v| tcx.intern_predicates(v))
    }
}

impl<'tcx, T: TypeFoldable<'tcx>, I: Idx> TypeFoldable<'tcx> for IndexVec<I, T> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, folder: &mut F) -> Result<Self, F::Error> {
        self.try_map_id(|x| x.try_fold_with(folder))
    }
}

impl<'tcx, T: TypeVisitable<'tcx>, I: Idx> TypeVisitable<'tcx> for IndexVec<I, T> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::Const<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, folder: &mut F) -> Result<Self, F::Error> {
        folder.try_fold_const(self)
    }
}

impl<'tcx> TypeVisitable<'tcx> for ty::Const<'tcx> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        visitor.visit_const(*self)
    }
}

impl<'tcx> TypeSuperFoldable<'tcx> for ty::Const<'tcx> {
    fn try_super_fold_with<F: FallibleTypeFolder<'tcx>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        let ty = self.ty().try_fold_with(folder)?;
        let kind = self.kind().try_fold_with(folder)?;
        if ty != self.ty() || kind != self.kind() {
            Ok(folder.tcx().mk_const(kind, ty))
        } else {
            Ok(self)
        }
    }
}

impl<'tcx> TypeSuperVisitable<'tcx> for ty::Const<'tcx> {
    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.ty().visit_with(visitor)?;
        self.kind().visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for InferConst<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, _folder: &mut F) -> Result<Self, F::Error> {
        Ok(self)
    }
}

impl<'tcx> TypeVisitable<'tcx> for InferConst<'tcx> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, _visitor: &mut V) -> ControlFlow<V::BreakTy> {
        ControlFlow::Continue(())
    }
}

impl<'tcx> TypeSuperVisitable<'tcx> for ty::UnevaluatedConst<'tcx> {
    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.substs.visit_with(visitor)
    }
}

impl<'tcx> TypeVisitable<'tcx> for TyAndLayout<'tcx, Ty<'tcx>> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        visitor.visit_ty(self.ty)
    }
}
