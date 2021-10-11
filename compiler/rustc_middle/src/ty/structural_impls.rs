//! This module contains implements of the `Lift` and `TypeFoldable`
//! traits for various types in the Rust compiler. Most are written by
//! hand, though we've recently added some macros and proc-macros to help with the tedium.

use crate::mir::interpret;
use crate::mir::ProjectionKind;
use crate::ty::fold::{TypeFoldable, TypeFolder, TypeVisitor};
use crate::ty::print::{with_no_trimmed_paths, FmtPrinter, Printer};
use crate::ty::{self, InferConst, Lift, Ty, TyCtxt};
use rustc_data_structures::functor::IdFunctor;
use rustc_hir::def::Namespace;
use rustc_hir::def_id::CRATE_DEF_INDEX;
use rustc_index::vec::{Idx, IndexVec};

use std::fmt;
use std::ops::ControlFlow;
use std::rc::Rc;
use std::sync::Arc;

impl fmt::Debug for ty::TraitDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ty::tls::with(|tcx| {
            with_no_trimmed_paths(|| {
                FmtPrinter::new(tcx, f, Namespace::TypeNS).print_def_path(self.def_id, &[])
            })?;
            Ok(())
        })
    }
}

impl fmt::Debug for ty::AdtDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ty::tls::with(|tcx| {
            with_no_trimmed_paths(|| {
                FmtPrinter::new(tcx, f, Namespace::TypeNS).print_def_path(self.did, &[])
            })?;
            Ok(())
        })
    }
}

impl fmt::Debug for ty::UpvarId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = ty::tls::with(|tcx| tcx.hir().name(self.var_path.hir_id));
        write!(f, "UpvarId({:?};`{}`;{:?})", self.var_path.hir_id, name, self.closure_expr_id)
    }
}

impl fmt::Debug for ty::UpvarBorrow<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UpvarBorrow({:?}, {:?})", self.kind, self.region)
    }
}

impl fmt::Debug for ty::ExistentialTraitRef<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        with_no_trimmed_paths(|| fmt::Display::fmt(self, f))
    }
}

impl fmt::Debug for ty::adjustment::Adjustment<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} -> {}", self.kind, self.target)
    }
}

impl fmt::Debug for ty::BoundRegionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ty::BrAnon(n) => write!(f, "BrAnon({:?})", n),
            ty::BrNamed(did, name) => {
                if did.index == CRATE_DEF_INDEX {
                    write!(f, "BrNamed({})", name)
                } else {
                    write!(f, "BrNamed({:?}, {})", did, name)
                }
            }
            ty::BrEnv => write!(f, "BrEnv"),
        }
    }
}

impl fmt::Debug for ty::RegionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ty::ReEarlyBound(ref data) => write!(f, "ReEarlyBound({}, {})", data.index, data.name),

            ty::ReLateBound(binder_id, ref bound_region) => {
                write!(f, "ReLateBound({:?}, {:?})", binder_id, bound_region)
            }

            ty::ReFree(ref fr) => fr.fmt(f),

            ty::ReStatic => write!(f, "ReStatic"),

            ty::ReVar(ref vid) => vid.fmt(f),

            ty::RePlaceholder(placeholder) => write!(f, "RePlaceholder({:?})", placeholder),

            ty::ReEmpty(ui) => write!(f, "ReEmpty({:?})", ui),

            ty::ReErased => write!(f, "ReErased"),
        }
    }
}

impl fmt::Debug for ty::FreeRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReFree({:?}, {:?})", self.scope, self.bound_region)
    }
}

impl fmt::Debug for ty::FnSig<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:?}; c_variadic: {})->{:?}", self.inputs(), self.c_variadic, self.output())
    }
}

impl<'tcx> fmt::Debug for ty::ConstVid<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_#{}c", self.index)
    }
}

impl fmt::Debug for ty::RegionVid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "'_#{}r", self.index())
    }
}

impl fmt::Debug for ty::TraitRef<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        with_no_trimmed_paths(|| fmt::Display::fmt(self, f))
    }
}

impl fmt::Debug for Ty<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        with_no_trimmed_paths(|| fmt::Display::fmt(self, f))
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

impl fmt::Debug for ty::TraitPredicate<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let ty::BoundConstness::ConstIfConst = self.constness {
            write!(f, "~const ")?;
        }
        write!(f, "TraitPredicate({:?}, polarity:{:?})", self.trait_ref, self.polarity)
    }
}

impl fmt::Debug for ty::ProjectionPredicate<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ProjectionPredicate({:?}, {:?})", self.projection_ty, self.ty)
    }
}

impl fmt::Debug for ty::Predicate<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.kind())
    }
}

impl fmt::Debug for ty::PredicateKind<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ty::PredicateKind::Trait(ref a) => a.fmt(f),
            ty::PredicateKind::Subtype(ref pair) => pair.fmt(f),
            ty::PredicateKind::Coerce(ref pair) => pair.fmt(f),
            ty::PredicateKind::RegionOutlives(ref pair) => pair.fmt(f),
            ty::PredicateKind::TypeOutlives(ref pair) => pair.fmt(f),
            ty::PredicateKind::Projection(ref pair) => pair.fmt(f),
            ty::PredicateKind::WellFormed(data) => write!(f, "WellFormed({:?})", data),
            ty::PredicateKind::ObjectSafe(trait_def_id) => {
                write!(f, "ObjectSafe({:?})", trait_def_id)
            }
            ty::PredicateKind::ClosureKind(closure_def_id, closure_substs, kind) => {
                write!(f, "ClosureKind({:?}, {:?}, {:?})", closure_def_id, closure_substs, kind)
            }
            ty::PredicateKind::ConstEvaluatable(uv) => {
                write!(f, "ConstEvaluatable({:?}, {:?})", uv.def, uv.substs_)
            }
            ty::PredicateKind::ConstEquate(c1, c2) => write!(f, "ConstEquate({:?}, {:?})", c1, c2),
            ty::PredicateKind::TypeWellFormedFromEnv(ty) => {
                write!(f, "TypeWellFormedFromEnv({:?})", ty)
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Atomic structs
//
// For things that don't carry any arena-allocated data (and are
// copy...), just add them to this list.

TrivialTypeFoldableAndLiftImpls! {
    (),
    bool,
    usize,
    ::rustc_target::abi::VariantIdx,
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
    ::rustc_hir::LlvmInlineAsmInner,
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
    crate::ty::Placeholder<crate::ty::BoundRegionKind>,
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
}

///////////////////////////////////////////////////////////////////////////
// Lift implementations

// FIXME(eddyb) replace all the uses of `Option::map` with `?`.
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
        match self {
            Some(x) => tcx.lift(x).map(Some),
            None => Some(None),
        }
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
        tcx.lift(*self).map(Box::new)
    }
}

impl<'tcx, T: Lift<'tcx> + Clone> Lift<'tcx> for Rc<T> {
    type Lifted = Rc<T::Lifted>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(self.as_ref().clone()).map(Rc::new)
    }
}

impl<'tcx, T: Lift<'tcx> + Clone> Lift<'tcx> for Arc<T> {
    type Lifted = Arc<T::Lifted>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(self.as_ref().clone()).map(Arc::new)
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

impl<'a, 'tcx> Lift<'tcx> for ty::TraitRef<'a> {
    type Lifted = ty::TraitRef<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(self.substs).map(|substs| ty::TraitRef { def_id: self.def_id, substs })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ExistentialTraitRef<'a> {
    type Lifted = ty::ExistentialTraitRef<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(self.substs).map(|substs| ty::ExistentialTraitRef { def_id: self.def_id, substs })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ExistentialPredicate<'a> {
    type Lifted = ty::ExistentialPredicate<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        match self {
            ty::ExistentialPredicate::Trait(x) => tcx.lift(x).map(ty::ExistentialPredicate::Trait),
            ty::ExistentialPredicate::Projection(x) => {
                tcx.lift(x).map(ty::ExistentialPredicate::Projection)
            }
            ty::ExistentialPredicate::AutoTrait(def_id) => {
                Some(ty::ExistentialPredicate::AutoTrait(def_id))
            }
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::TraitPredicate<'a> {
    type Lifted = ty::TraitPredicate<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<ty::TraitPredicate<'tcx>> {
        tcx.lift(self.trait_ref).map(|trait_ref| ty::TraitPredicate {
            trait_ref,
            constness: self.constness,
            polarity: self.polarity,
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::SubtypePredicate<'a> {
    type Lifted = ty::SubtypePredicate<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<ty::SubtypePredicate<'tcx>> {
        tcx.lift((self.a, self.b)).map(|(a, b)| ty::SubtypePredicate {
            a_is_expected: self.a_is_expected,
            a,
            b,
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::CoercePredicate<'a> {
    type Lifted = ty::CoercePredicate<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<ty::CoercePredicate<'tcx>> {
        tcx.lift((self.a, self.b)).map(|(a, b)| ty::CoercePredicate { a, b })
    }
}

impl<'tcx, A: Copy + Lift<'tcx>, B: Copy + Lift<'tcx>> Lift<'tcx> for ty::OutlivesPredicate<A, B> {
    type Lifted = ty::OutlivesPredicate<A::Lifted, B::Lifted>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift((self.0, self.1)).map(|(a, b)| ty::OutlivesPredicate(a, b))
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ProjectionTy<'a> {
    type Lifted = ty::ProjectionTy<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<ty::ProjectionTy<'tcx>> {
        tcx.lift(self.substs)
            .map(|substs| ty::ProjectionTy { item_def_id: self.item_def_id, substs })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ProjectionPredicate<'a> {
    type Lifted = ty::ProjectionPredicate<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<ty::ProjectionPredicate<'tcx>> {
        tcx.lift((self.projection_ty, self.ty))
            .map(|(projection_ty, ty)| ty::ProjectionPredicate { projection_ty, ty })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ExistentialProjection<'a> {
    type Lifted = ty::ExistentialProjection<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(self.substs).map(|substs| ty::ExistentialProjection {
            substs,
            ty: tcx.lift(self.ty).expect("type must lift when substs do"),
            item_def_id: self.item_def_id,
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::PredicateKind<'a> {
    type Lifted = ty::PredicateKind<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        match self {
            ty::PredicateKind::Trait(data) => tcx.lift(data).map(ty::PredicateKind::Trait),
            ty::PredicateKind::Subtype(data) => tcx.lift(data).map(ty::PredicateKind::Subtype),
            ty::PredicateKind::Coerce(data) => tcx.lift(data).map(ty::PredicateKind::Coerce),
            ty::PredicateKind::RegionOutlives(data) => {
                tcx.lift(data).map(ty::PredicateKind::RegionOutlives)
            }
            ty::PredicateKind::TypeOutlives(data) => {
                tcx.lift(data).map(ty::PredicateKind::TypeOutlives)
            }
            ty::PredicateKind::Projection(data) => {
                tcx.lift(data).map(ty::PredicateKind::Projection)
            }
            ty::PredicateKind::WellFormed(ty) => tcx.lift(ty).map(ty::PredicateKind::WellFormed),
            ty::PredicateKind::ClosureKind(closure_def_id, closure_substs, kind) => {
                tcx.lift(closure_substs).map(|closure_substs| {
                    ty::PredicateKind::ClosureKind(closure_def_id, closure_substs, kind)
                })
            }
            ty::PredicateKind::ObjectSafe(trait_def_id) => {
                Some(ty::PredicateKind::ObjectSafe(trait_def_id))
            }
            ty::PredicateKind::ConstEvaluatable(uv) => {
                tcx.lift(uv).map(|uv| ty::PredicateKind::ConstEvaluatable(uv))
            }
            ty::PredicateKind::ConstEquate(c1, c2) => {
                tcx.lift((c1, c2)).map(|(c1, c2)| ty::PredicateKind::ConstEquate(c1, c2))
            }
            ty::PredicateKind::TypeWellFormedFromEnv(ty) => {
                tcx.lift(ty).map(ty::PredicateKind::TypeWellFormedFromEnv)
            }
        }
    }
}

impl<'a, 'tcx, T: Lift<'tcx>> Lift<'tcx> for ty::Binder<'a, T>
where
    <T as Lift<'tcx>>::Lifted: TypeFoldable<'tcx>,
{
    type Lifted = ty::Binder<'tcx, T::Lifted>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        let bound_vars = tcx.lift(self.bound_vars());
        tcx.lift(self.skip_binder())
            .zip(bound_vars)
            .map(|(value, vars)| ty::Binder::bind_with_vars(value, vars))
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ParamEnv<'a> {
    type Lifted = ty::ParamEnv<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(self.caller_bounds())
            .map(|caller_bounds| ty::ParamEnv::new(caller_bounds, self.reveal()))
    }
}

impl<'a, 'tcx, T: Lift<'tcx>> Lift<'tcx> for ty::ParamEnvAnd<'a, T> {
    type Lifted = ty::ParamEnvAnd<'tcx, T::Lifted>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(self.param_env).and_then(|param_env| {
            tcx.lift(self.value).map(|value| ty::ParamEnvAnd { param_env, value })
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ClosureSubsts<'a> {
    type Lifted = ty::ClosureSubsts<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(self.substs).map(|substs| ty::ClosureSubsts { substs })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::GeneratorSubsts<'a> {
    type Lifted = ty::GeneratorSubsts<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(self.substs).map(|substs| ty::GeneratorSubsts { substs })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::adjustment::Adjustment<'a> {
    type Lifted = ty::adjustment::Adjustment<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        let ty::adjustment::Adjustment { kind, target } = self;
        tcx.lift(kind).and_then(|kind| {
            tcx.lift(target).map(|target| ty::adjustment::Adjustment { kind, target })
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::adjustment::Adjust<'a> {
    type Lifted = ty::adjustment::Adjust<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        match self {
            ty::adjustment::Adjust::NeverToAny => Some(ty::adjustment::Adjust::NeverToAny),
            ty::adjustment::Adjust::Pointer(ptr) => Some(ty::adjustment::Adjust::Pointer(ptr)),
            ty::adjustment::Adjust::Deref(overloaded) => {
                tcx.lift(overloaded).map(ty::adjustment::Adjust::Deref)
            }
            ty::adjustment::Adjust::Borrow(autoref) => {
                tcx.lift(autoref).map(ty::adjustment::Adjust::Borrow)
            }
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::adjustment::OverloadedDeref<'a> {
    type Lifted = ty::adjustment::OverloadedDeref<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(self.region).map(|region| ty::adjustment::OverloadedDeref {
            region,
            mutbl: self.mutbl,
            span: self.span,
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::adjustment::AutoBorrow<'a> {
    type Lifted = ty::adjustment::AutoBorrow<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        match self {
            ty::adjustment::AutoBorrow::Ref(r, m) => {
                tcx.lift(r).map(|r| ty::adjustment::AutoBorrow::Ref(r, m))
            }
            ty::adjustment::AutoBorrow::RawPtr(m) => Some(ty::adjustment::AutoBorrow::RawPtr(m)),
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::GenSig<'a> {
    type Lifted = ty::GenSig<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift((self.resume_ty, self.yield_ty, self.return_ty))
            .map(|(resume_ty, yield_ty, return_ty)| ty::GenSig { resume_ty, yield_ty, return_ty })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::FnSig<'a> {
    type Lifted = ty::FnSig<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(self.inputs_and_output).map(|x| ty::FnSig {
            inputs_and_output: x,
            c_variadic: self.c_variadic,
            unsafety: self.unsafety,
            abi: self.abi,
        })
    }
}

impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for ty::error::ExpectedFound<T> {
    type Lifted = ty::error::ExpectedFound<T::Lifted>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        let ty::error::ExpectedFound { expected, found } = self;
        tcx.lift(expected).and_then(|expected| {
            tcx.lift(found).map(|found| ty::error::ExpectedFound { expected, found })
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::error::TypeError<'a> {
    type Lifted = ty::error::TypeError<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        use crate::ty::error::TypeError::*;

        Some(match self {
            Mismatch => Mismatch,
            ConstnessMismatch(x) => ConstnessMismatch(x),
            PolarityMismatch(x) => PolarityMismatch(x),
            UnsafetyMismatch(x) => UnsafetyMismatch(x),
            AbiMismatch(x) => AbiMismatch(x),
            Mutability => Mutability,
            ArgumentMutability(i) => ArgumentMutability(i),
            TupleSize(x) => TupleSize(x),
            FixedArraySize(x) => FixedArraySize(x),
            ArgCount => ArgCount,
            RegionsDoesNotOutlive(a, b) => {
                return tcx.lift((a, b)).map(|(a, b)| RegionsDoesNotOutlive(a, b));
            }
            RegionsInsufficientlyPolymorphic(a, b) => {
                return tcx.lift(b).map(|b| RegionsInsufficientlyPolymorphic(a, b));
            }
            RegionsOverlyPolymorphic(a, b) => {
                return tcx.lift(b).map(|b| RegionsOverlyPolymorphic(a, b));
            }
            RegionsPlaceholderMismatch => RegionsPlaceholderMismatch,
            IntMismatch(x) => IntMismatch(x),
            FloatMismatch(x) => FloatMismatch(x),
            Traits(x) => Traits(x),
            VariadicMismatch(x) => VariadicMismatch(x),
            CyclicTy(t) => return tcx.lift(t).map(|t| CyclicTy(t)),
            CyclicConst(ct) => return tcx.lift(ct).map(|ct| CyclicConst(ct)),
            ProjectionMismatched(x) => ProjectionMismatched(x),
            ArgumentSorts(x, i) => return tcx.lift(x).map(|x| ArgumentSorts(x, i)),
            Sorts(x) => return tcx.lift(x).map(Sorts),
            ExistentialMismatch(x) => return tcx.lift(x).map(ExistentialMismatch),
            ConstMismatch(x) => return tcx.lift(x).map(ConstMismatch),
            IntrinsicCast => IntrinsicCast,
            TargetFeatureCast(x) => TargetFeatureCast(x),
            ObjectUnsafeCoercion(x) => return tcx.lift(x).map(ObjectUnsafeCoercion),
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::InstanceDef<'a> {
    type Lifted = ty::InstanceDef<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        match self {
            ty::InstanceDef::Item(def_id) => Some(ty::InstanceDef::Item(def_id)),
            ty::InstanceDef::VtableShim(def_id) => Some(ty::InstanceDef::VtableShim(def_id)),
            ty::InstanceDef::ReifyShim(def_id) => Some(ty::InstanceDef::ReifyShim(def_id)),
            ty::InstanceDef::Intrinsic(def_id) => Some(ty::InstanceDef::Intrinsic(def_id)),
            ty::InstanceDef::FnPtrShim(def_id, ty) => {
                Some(ty::InstanceDef::FnPtrShim(def_id, tcx.lift(ty)?))
            }
            ty::InstanceDef::Virtual(def_id, n) => Some(ty::InstanceDef::Virtual(def_id, n)),
            ty::InstanceDef::ClosureOnceShim { call_once, track_caller } => {
                Some(ty::InstanceDef::ClosureOnceShim { call_once, track_caller })
            }
            ty::InstanceDef::DropGlue(def_id, ty) => {
                Some(ty::InstanceDef::DropGlue(def_id, tcx.lift(ty)?))
            }
            ty::InstanceDef::CloneShim(def_id, ty) => {
                Some(ty::InstanceDef::CloneShim(def_id, tcx.lift(ty)?))
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// TypeFoldable implementations.
//
// Ideally, each type should invoke `folder.fold_foo(self)` and
// nothing else. In some cases, though, we haven't gotten around to
// adding methods on the `folder` yet, and thus the folding is
// hard-coded here. This is less-flexible, because folders cannot
// override the behavior, but there are a lot of random types and one
// can easily refactor the folding into the TypeFolder trait as
// needed.

/// AdtDefs are basically the same as a DefId.
impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::AdtDef {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, _folder: &mut F) -> Self {
        self
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, _visitor: &mut V) -> ControlFlow<V::BreakTy> {
        ControlFlow::CONTINUE
    }
}

impl<'tcx, T: TypeFoldable<'tcx>, U: TypeFoldable<'tcx>> TypeFoldable<'tcx> for (T, U) {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> (T, U) {
        (self.0.fold_with(folder), self.1.fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.0.visit_with(visitor)?;
        self.1.visit_with(visitor)
    }
}

impl<'tcx, A: TypeFoldable<'tcx>, B: TypeFoldable<'tcx>, C: TypeFoldable<'tcx>> TypeFoldable<'tcx>
    for (A, B, C)
{
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> (A, B, C) {
        (self.0.fold_with(folder), self.1.fold_with(folder), self.2.fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.0.visit_with(visitor)?;
        self.1.visit_with(visitor)?;
        self.2.visit_with(visitor)
    }
}

EnumTypeFoldableImpl! {
    impl<'tcx, T> TypeFoldable<'tcx> for Option<T> {
        (Some)(a),
        (None),
    } where T: TypeFoldable<'tcx>
}

EnumTypeFoldableImpl! {
    impl<'tcx, T, E> TypeFoldable<'tcx> for Result<T, E> {
        (Ok)(a),
        (Err)(a),
    } where T: TypeFoldable<'tcx>, E: TypeFoldable<'tcx>,
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Rc<T> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        // FIXME: Reuse the `Rc` here.
        Rc::new((*self).clone().fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        (**self).visit_with(visitor)
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Arc<T> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        // FIXME: Reuse the `Arc` here.
        Arc::new((*self).clone().fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        (**self).visit_with(visitor)
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Box<T> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        self.map_id(|value| value.fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        (**self).visit_with(visitor)
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Vec<T> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        self.map_id(|t| t.fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Box<[T]> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        self.map_id(|t| t.fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for ty::Binder<'tcx, T> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        self.map_bound(|ty| ty.fold_with(folder))
    }

    fn fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        folder.fold_binder(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.as_ref().skip_binder().visit_with(visitor)
    }

    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        visitor.visit_binder(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::List<ty::Binder<'tcx, ty::ExistentialPredicate<'tcx>>> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        ty::util::fold_list(self, folder, |tcx, v| tcx.intern_poly_existential_predicates(v))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|p| p.visit_with(visitor))
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::List<Ty<'tcx>> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        ty::util::fold_list(self, folder, |tcx, v| tcx.intern_type_list(v))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::List<ProjectionKind> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        ty::util::fold_list(self, folder, |tcx, v| tcx.intern_projs(v))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::instance::Instance<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        use crate::ty::InstanceDef::*;
        Self {
            substs: self.substs.fold_with(folder),
            def: match self.def {
                Item(def) => Item(def.fold_with(folder)),
                VtableShim(did) => VtableShim(did.fold_with(folder)),
                ReifyShim(did) => ReifyShim(did.fold_with(folder)),
                Intrinsic(did) => Intrinsic(did.fold_with(folder)),
                FnPtrShim(did, ty) => FnPtrShim(did.fold_with(folder), ty.fold_with(folder)),
                Virtual(did, i) => Virtual(did.fold_with(folder), i),
                ClosureOnceShim { call_once, track_caller } => {
                    ClosureOnceShim { call_once: call_once.fold_with(folder), track_caller }
                }
                DropGlue(did, ty) => DropGlue(did.fold_with(folder), ty.fold_with(folder)),
                CloneShim(did, ty) => CloneShim(did.fold_with(folder), ty.fold_with(folder)),
            },
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        use crate::ty::InstanceDef::*;
        self.substs.visit_with(visitor)?;
        match self.def {
            Item(def) => def.visit_with(visitor),
            VtableShim(did) | ReifyShim(did) | Intrinsic(did) | Virtual(did, _) => {
                did.visit_with(visitor)
            }
            FnPtrShim(did, ty) | CloneShim(did, ty) => {
                did.visit_with(visitor)?;
                ty.visit_with(visitor)
            }
            DropGlue(did, ty) => {
                did.visit_with(visitor)?;
                ty.visit_with(visitor)
            }
            ClosureOnceShim { call_once, track_caller: _ } => call_once.visit_with(visitor),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for interpret::GlobalId<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        Self { instance: self.instance.fold_with(folder), promoted: self.promoted }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.instance.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for Ty<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        let kind = match *self.kind() {
            ty::RawPtr(tm) => ty::RawPtr(tm.fold_with(folder)),
            ty::Array(typ, sz) => ty::Array(typ.fold_with(folder), sz.fold_with(folder)),
            ty::Slice(typ) => ty::Slice(typ.fold_with(folder)),
            ty::Adt(tid, substs) => ty::Adt(tid, substs.fold_with(folder)),
            ty::Dynamic(trait_ty, region) => {
                ty::Dynamic(trait_ty.fold_with(folder), region.fold_with(folder))
            }
            ty::Tuple(ts) => ty::Tuple(ts.fold_with(folder)),
            ty::FnDef(def_id, substs) => ty::FnDef(def_id, substs.fold_with(folder)),
            ty::FnPtr(f) => ty::FnPtr(f.fold_with(folder)),
            ty::Ref(r, ty, mutbl) => ty::Ref(r.fold_with(folder), ty.fold_with(folder), mutbl),
            ty::Generator(did, substs, movability) => {
                ty::Generator(did, substs.fold_with(folder), movability)
            }
            ty::GeneratorWitness(types) => ty::GeneratorWitness(types.fold_with(folder)),
            ty::Closure(did, substs) => ty::Closure(did, substs.fold_with(folder)),
            ty::Projection(data) => ty::Projection(data.fold_with(folder)),
            ty::Opaque(did, substs) => ty::Opaque(did, substs.fold_with(folder)),

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
            | ty::Foreign(..) => return self,
        };

        if *self.kind() == kind { self } else { folder.tcx().mk_ty(kind) }
    }

    fn fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        folder.fold_ty(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        match self.kind() {
            ty::RawPtr(ref tm) => tm.visit_with(visitor),
            ty::Array(typ, sz) => {
                typ.visit_with(visitor)?;
                sz.visit_with(visitor)
            }
            ty::Slice(typ) => typ.visit_with(visitor),
            ty::Adt(_, substs) => substs.visit_with(visitor),
            ty::Dynamic(ref trait_ty, ref reg) => {
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
            ty::Closure(_did, ref substs) => substs.visit_with(visitor),
            ty::Projection(ref data) => data.visit_with(visitor),
            ty::Opaque(_, ref substs) => substs.visit_with(visitor),

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
            | ty::Foreign(..) => ControlFlow::CONTINUE,
        }
    }

    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        visitor.visit_ty(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::Region<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, _folder: &mut F) -> Self {
        self
    }

    fn fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        folder.fold_region(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, _visitor: &mut V) -> ControlFlow<V::BreakTy> {
        ControlFlow::CONTINUE
    }

    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        visitor.visit_region(*self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::Predicate<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        folder.fold_predicate(self)
    }

    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        let new = self.inner.kind.fold_with(folder);
        folder.tcx().reuse_or_mk_predicate(self, new)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.inner.kind.visit_with(visitor)
    }

    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        visitor.visit_predicate(*self)
    }

    fn has_vars_bound_at_or_above(&self, binder: ty::DebruijnIndex) -> bool {
        self.inner.outer_exclusive_binder > binder
    }

    fn has_type_flags(&self, flags: ty::TypeFlags) -> bool {
        self.inner.flags.intersects(flags)
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::List<ty::Predicate<'tcx>> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        ty::util::fold_list(self, folder, |tcx, v| tcx.intern_predicates(v))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|p| p.visit_with(visitor))
    }
}

impl<'tcx, T: TypeFoldable<'tcx>, I: Idx> TypeFoldable<'tcx> for IndexVec<I, T> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        self.map_id(|x| x.fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::Const<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        let ty = self.ty.fold_with(folder);
        let val = self.val.fold_with(folder);
        if ty != self.ty || val != self.val {
            folder.tcx().mk_const(ty::Const { ty, val })
        } else {
            self
        }
    }

    fn fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        folder.fold_const(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.ty.visit_with(visitor)?;
        self.val.visit_with(visitor)
    }

    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        visitor.visit_const(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ConstKind<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        match self {
            ty::ConstKind::Infer(ic) => ty::ConstKind::Infer(ic.fold_with(folder)),
            ty::ConstKind::Param(p) => ty::ConstKind::Param(p.fold_with(folder)),
            ty::ConstKind::Unevaluated(uv) => ty::ConstKind::Unevaluated(uv.fold_with(folder)),
            ty::ConstKind::Value(_)
            | ty::ConstKind::Bound(..)
            | ty::ConstKind::Placeholder(..)
            | ty::ConstKind::Error(_) => self,
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        match *self {
            ty::ConstKind::Infer(ic) => ic.visit_with(visitor),
            ty::ConstKind::Param(p) => p.visit_with(visitor),
            ty::ConstKind::Unevaluated(uv) => uv.visit_with(visitor),
            ty::ConstKind::Value(_)
            | ty::ConstKind::Bound(..)
            | ty::ConstKind::Placeholder(_)
            | ty::ConstKind::Error(_) => ControlFlow::CONTINUE,
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for InferConst<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, _folder: &mut F) -> Self {
        self
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, _visitor: &mut V) -> ControlFlow<V::BreakTy> {
        ControlFlow::CONTINUE
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::Unevaluated<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        ty::Unevaluated {
            def: self.def,
            substs_: Some(self.substs(folder.tcx()).fold_with(folder)),
            promoted: self.promoted,
        }
    }

    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        visitor.visit_unevaluated_const(*self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        if let Some(tcx) = visitor.tcx_for_anon_const_substs() {
            self.substs(tcx).visit_with(visitor)
        } else if let Some(substs) = self.substs_ {
            substs.visit_with(visitor)
        } else {
            debug!("ignoring default substs of `{:?}`", self.def);
            ControlFlow::CONTINUE
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::Unevaluated<'tcx, ()> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        ty::Unevaluated {
            def: self.def,
            substs_: Some(self.substs(folder.tcx()).fold_with(folder)),
            promoted: self.promoted,
        }
    }

    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        visitor.visit_unevaluated_const(self.expand())
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        if let Some(tcx) = visitor.tcx_for_anon_const_substs() {
            self.substs(tcx).visit_with(visitor)
        } else if let Some(substs) = self.substs_ {
            substs.visit_with(visitor)
        } else {
            debug!("ignoring default substs of `{:?}`", self.def);
            ControlFlow::CONTINUE
        }
    }
}
