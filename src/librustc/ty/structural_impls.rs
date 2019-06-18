//! This module contains implements of the `Lift` and `TypeFoldable`
//! traits for various types in the Rust compiler. Most are written by
//! hand, though we've recently added some macros (e.g.,
//! `BraceStructLiftImpl!`) to help with the tedium.

use crate::hir::def::Namespace;
use crate::mir::ProjectionKind;
use crate::mir::interpret::ConstValue;
use crate::ty::{self, Lift, Ty, TyCtxt, InferConst};
use crate::ty::fold::{TypeFoldable, TypeFolder, TypeVisitor};
use crate::ty::print::{FmtPrinter, Printer};
use rustc_data_structures::indexed_vec::{IndexVec, Idx};
use smallvec::SmallVec;
use crate::mir::interpret;

use std::fmt;
use std::rc::Rc;
use std::sync::Arc;

impl fmt::Debug for ty::GenericParamDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let type_name = match self.kind {
            ty::GenericParamDefKind::Lifetime => "Lifetime",
            ty::GenericParamDefKind::Type {..} => "Type",
            ty::GenericParamDefKind::Const => "Const",
        };
        write!(f, "{}({}, {:?}, {})",
               type_name,
               self.name,
               self.def_id,
               self.index)
    }
}

impl fmt::Debug for ty::TraitDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ty::tls::with(|tcx| {
            FmtPrinter::new(tcx, f, Namespace::TypeNS)
                .print_def_path(self.def_id, &[])?;
            Ok(())
        })
    }
}

impl fmt::Debug for ty::AdtDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ty::tls::with(|tcx| {
            FmtPrinter::new(tcx, f, Namespace::TypeNS)
                .print_def_path(self.did, &[])?;
            Ok(())
        })
    }
}

impl fmt::Debug for ty::ClosureUpvar<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ClosureUpvar({:?},{:?})",
               self.res,
               self.ty)
    }
}

impl fmt::Debug for ty::UpvarId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = ty::tls::with(|tcx| {
            tcx.hir().name(self.var_path.hir_id)
        });
        write!(f, "UpvarId({:?};`{}`;{:?})",
            self.var_path.hir_id,
            name,
            self.closure_expr_id)
    }
}

impl fmt::Debug for ty::UpvarBorrow<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UpvarBorrow({:?}, {:?})",
               self.kind, self.region)
    }
}

impl fmt::Debug for ty::ExistentialTraitRef<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Debug for ty::adjustment::Adjustment<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} -> {}", self.kind, self.target)
    }
}

impl fmt::Debug for ty::BoundRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ty::BrAnon(n) => write!(f, "BrAnon({:?})", n),
            ty::BrNamed(did, name) => {
                write!(f, "BrNamed({:?}:{:?}, {})",
                        did.krate, did.index, name)
            }
            ty::BrEnv => write!(f, "BrEnv"),
        }
    }
}

impl fmt::Debug for ty::RegionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ty::ReEarlyBound(ref data) => {
                write!(f, "ReEarlyBound({}, {})",
                        data.index,
                        data.name)
            }

            ty::ReClosureBound(ref vid) => {
                write!(f, "ReClosureBound({:?})", vid)
            }

            ty::ReLateBound(binder_id, ref bound_region) => {
                write!(f, "ReLateBound({:?}, {:?})", binder_id, bound_region)
            }

            ty::ReFree(ref fr) => fr.fmt(f),

            ty::ReScope(id) => write!(f, "ReScope({:?})", id),

            ty::ReStatic => write!(f, "ReStatic"),

            ty::ReVar(ref vid) => vid.fmt(f),

            ty::RePlaceholder(placeholder) => {
                write!(f, "RePlaceholder({:?})", placeholder)
            }

            ty::ReEmpty => write!(f, "ReEmpty"),

            ty::ReErased => write!(f, "ReErased"),
        }
    }
}

impl fmt::Debug for ty::FreeRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReFree({:?}, {:?})", self.scope, self.bound_region)
    }
}

impl fmt::Debug for ty::Variance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match *self {
            ty::Covariant => "+",
            ty::Contravariant => "-",
            ty::Invariant => "o",
            ty::Bivariant => "*",
        })
    }
}

impl fmt::Debug for ty::FnSig<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:?}; c_variadic: {})->{:?}",
                self.inputs(), self.c_variadic, self.output())
    }
}

impl fmt::Debug for ty::TyVid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_#{}t", self.index)
    }
}

impl<'tcx> fmt::Debug for ty::ConstVid<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_#{}c", self.index)
    }
}

impl fmt::Debug for ty::IntVid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_#{}i", self.index)
    }
}

impl fmt::Debug for ty::FloatVid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_#{}f", self.index)
    }
}

impl fmt::Debug for ty::RegionVid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "'_#{}r", self.index())
    }
}

impl fmt::Debug for ty::InferTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ty::TyVar(ref v) => v.fmt(f),
            ty::IntVar(ref v) => v.fmt(f),
            ty::FloatVar(ref v) => v.fmt(f),
            ty::FreshTy(v) => write!(f, "FreshTy({:?})", v),
            ty::FreshIntTy(v) => write!(f, "FreshIntTy({:?})", v),
            ty::FreshFloatTy(v) => write!(f, "FreshFloatTy({:?})", v),
        }
    }
}

impl fmt::Debug for ty::IntVarValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ty::IntType(ref v) => v.fmt(f),
            ty::UintType(ref v) => v.fmt(f),
        }
    }
}

impl fmt::Debug for ty::FloatVarValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl fmt::Debug for ty::TraitRef<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // FIXME(#59188) this is used across the compiler to print
        // a `TraitRef` qualified (with the Self type explicit),
        // instead of having a different way to make that choice.
        write!(f, "<{} as {}>", self.self_ty(), self)
    }
}

impl fmt::Debug for Ty<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
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
        write!(f, "TraitPredicate({:?})", self.trait_ref)
    }
}

impl fmt::Debug for ty::ProjectionPredicate<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ProjectionPredicate({:?}, {:?})", self.projection_ty, self.ty)
    }
}

impl fmt::Debug for ty::Predicate<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ty::Predicate::Trait(ref a) => a.fmt(f),
            ty::Predicate::Subtype(ref pair) => pair.fmt(f),
            ty::Predicate::RegionOutlives(ref pair) => pair.fmt(f),
            ty::Predicate::TypeOutlives(ref pair) => pair.fmt(f),
            ty::Predicate::Projection(ref pair) => pair.fmt(f),
            ty::Predicate::WellFormed(ty) => write!(f, "WellFormed({:?})", ty),
            ty::Predicate::ObjectSafe(trait_def_id) => {
                write!(f, "ObjectSafe({:?})", trait_def_id)
            }
            ty::Predicate::ClosureKind(closure_def_id, closure_substs, kind) => {
                write!(f, "ClosureKind({:?}, {:?}, {:?})",
                    closure_def_id, closure_substs, kind)
            }
            ty::Predicate::ConstEvaluatable(def_id, substs) => {
                write!(f, "ConstEvaluatable({:?}, {:?})", def_id, substs)
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Atomic structs
//
// For things that don't carry any arena-allocated data (and are
// copy...), just add them to this list.

CloneTypeFoldableAndLiftImpls! {
    (),
    bool,
    usize,
    crate::ty::layout::VariantIdx,
    u64,
    String,
    crate::middle::region::Scope,
    ::syntax::ast::FloatTy,
    ::syntax::ast::NodeId,
    ::syntax_pos::symbol::Symbol,
    crate::hir::def::Res,
    crate::hir::def_id::DefId,
    crate::hir::InlineAsm,
    crate::hir::MatchSource,
    crate::hir::Mutability,
    crate::hir::Unsafety,
    ::rustc_target::spec::abi::Abi,
    crate::mir::Local,
    crate::mir::Promoted,
    crate::traits::Reveal,
    crate::ty::adjustment::AutoBorrowMutability,
    crate::ty::AdtKind,
    // Including `BoundRegion` is a *bit* dubious, but direct
    // references to bound region appear in `ty::Error`, and aren't
    // really meant to be folded. In general, we can only fold a fully
    // general `Region`.
    crate::ty::BoundRegion,
    crate::ty::Placeholder<crate::ty::BoundRegion>,
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
    ::syntax_pos::Span,
}

///////////////////////////////////////////////////////////////////////////
// Lift implementations

// FIXME(eddyb) replace all the uses of `Option::map` with `?`.
impl<'tcx, A: Lift<'tcx>, B: Lift<'tcx>> Lift<'tcx> for (A, B) {
    type Lifted = (A::Lifted, B::Lifted);
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.0).and_then(|a| tcx.lift(&self.1).map(|b| (a, b)))
    }
}

impl<'tcx, A: Lift<'tcx>, B: Lift<'tcx>, C: Lift<'tcx>> Lift<'tcx> for (A, B, C) {
    type Lifted = (A::Lifted, B::Lifted, C::Lifted);
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.0).and_then(|a| {
            tcx.lift(&self.1).and_then(|b| tcx.lift(&self.2).map(|c| (a, b, c)))
        })
   }
}

impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for Option<T> {
    type Lifted = Option<T::Lifted>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        match *self {
            Some(ref x) => tcx.lift(x).map(Some),
            None => Some(None)
        }
    }
}

impl<'tcx, T: Lift<'tcx>, E: Lift<'tcx>> Lift<'tcx> for Result<T, E> {
    type Lifted = Result<T::Lifted, E::Lifted>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        match *self {
            Ok(ref x) => tcx.lift(x).map(Ok),
            Err(ref e) => tcx.lift(e).map(Err)
        }
    }
}

impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for Box<T> {
    type Lifted = Box<T::Lifted>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&**self).map(Box::new)
    }
}

impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for Rc<T> {
    type Lifted = Rc<T::Lifted>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&**self).map(Rc::new)
    }
}

impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for Arc<T> {
    type Lifted = Arc<T::Lifted>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&**self).map(Arc::new)
    }
}

impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for [T] {
    type Lifted = Vec<T::Lifted>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        // type annotation needed to inform `projection_must_outlive`
        let mut result : Vec<<T as Lift<'tcx>>::Lifted>
            = Vec::with_capacity(self.len());
        for x in self {
            if let Some(value) = tcx.lift(x) {
                result.push(value);
            } else {
                return None;
            }
        }
        Some(result)
    }
}

impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for Vec<T> {
    type Lifted = Vec<T::Lifted>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self[..])
    }
}

impl<'tcx, I: Idx, T: Lift<'tcx>> Lift<'tcx> for IndexVec<I, T> {
    type Lifted = IndexVec<I, T::Lifted>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        self.iter()
            .map(|e| tcx.lift(e))
            .collect()
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::TraitRef<'a> {
    type Lifted = ty::TraitRef<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.substs).map(|substs| ty::TraitRef {
            def_id: self.def_id,
            substs,
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ExistentialTraitRef<'a> {
    type Lifted = ty::ExistentialTraitRef<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.substs).map(|substs| ty::ExistentialTraitRef {
            def_id: self.def_id,
            substs,
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ExistentialPredicate<'a> {
    type Lifted = ty::ExistentialPredicate<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        match self {
            ty::ExistentialPredicate::Trait(x) => {
                tcx.lift(x).map(ty::ExistentialPredicate::Trait)
            }
            ty::ExistentialPredicate::Projection(x) => {
                tcx.lift(x).map(ty::ExistentialPredicate::Projection)
            }
            ty::ExistentialPredicate::AutoTrait(def_id) => {
                Some(ty::ExistentialPredicate::AutoTrait(*def_id))
            }
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::TraitPredicate<'a> {
    type Lifted = ty::TraitPredicate<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<ty::TraitPredicate<'tcx>> {
        tcx.lift(&self.trait_ref).map(|trait_ref| ty::TraitPredicate {
            trait_ref,
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::SubtypePredicate<'a> {
    type Lifted = ty::SubtypePredicate<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<ty::SubtypePredicate<'tcx>> {
        tcx.lift(&(self.a, self.b)).map(|(a, b)| ty::SubtypePredicate {
            a_is_expected: self.a_is_expected,
            a,
            b,
        })
    }
}

impl<'tcx, A: Copy + Lift<'tcx>, B: Copy + Lift<'tcx>> Lift<'tcx> for ty::OutlivesPredicate<A, B> {
    type Lifted = ty::OutlivesPredicate<A::Lifted, B::Lifted>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&(self.0, self.1)).map(|(a, b)| ty::OutlivesPredicate(a, b))
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ProjectionTy<'a> {
    type Lifted = ty::ProjectionTy<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<ty::ProjectionTy<'tcx>> {
        tcx.lift(&self.substs).map(|substs| {
            ty::ProjectionTy {
                item_def_id: self.item_def_id,
                substs,
            }
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ProjectionPredicate<'a> {
    type Lifted = ty::ProjectionPredicate<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<ty::ProjectionPredicate<'tcx>> {
        tcx.lift(&(self.projection_ty, self.ty)).map(|(projection_ty, ty)| {
            ty::ProjectionPredicate {
                projection_ty,
                ty,
            }
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ExistentialProjection<'a> {
    type Lifted = ty::ExistentialProjection<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.substs).map(|substs| {
            ty::ExistentialProjection {
                substs,
                ty: tcx.lift(&self.ty).expect("type must lift when substs do"),
                item_def_id: self.item_def_id,
            }
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::Predicate<'a> {
    type Lifted = ty::Predicate<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        match *self {
            ty::Predicate::Trait(ref binder) => {
                tcx.lift(binder).map(ty::Predicate::Trait)
            }
            ty::Predicate::Subtype(ref binder) => {
                tcx.lift(binder).map(ty::Predicate::Subtype)
            }
            ty::Predicate::RegionOutlives(ref binder) => {
                tcx.lift(binder).map(ty::Predicate::RegionOutlives)
            }
            ty::Predicate::TypeOutlives(ref binder) => {
                tcx.lift(binder).map(ty::Predicate::TypeOutlives)
            }
            ty::Predicate::Projection(ref binder) => {
                tcx.lift(binder).map(ty::Predicate::Projection)
            }
            ty::Predicate::WellFormed(ty) => {
                tcx.lift(&ty).map(ty::Predicate::WellFormed)
            }
            ty::Predicate::ClosureKind(closure_def_id, closure_substs, kind) => {
                tcx.lift(&closure_substs)
                   .map(|closure_substs| ty::Predicate::ClosureKind(closure_def_id,
                                                                    closure_substs,
                                                                    kind))
            }
            ty::Predicate::ObjectSafe(trait_def_id) => {
                Some(ty::Predicate::ObjectSafe(trait_def_id))
            }
            ty::Predicate::ConstEvaluatable(def_id, substs) => {
                tcx.lift(&substs).map(|substs| {
                    ty::Predicate::ConstEvaluatable(def_id, substs)
                })
            }
        }
    }
}

impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for ty::Binder<T> {
    type Lifted = ty::Binder<T::Lifted>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(self.skip_binder()).map(ty::Binder::bind)
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ParamEnv<'a> {
    type Lifted = ty::ParamEnv<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.caller_bounds).map(|caller_bounds| {
            ty::ParamEnv {
                reveal: self.reveal,
                caller_bounds,
                def_id: self.def_id,
            }
        })
    }
}

impl<'a, 'tcx, T: Lift<'tcx>> Lift<'tcx> for ty::ParamEnvAnd<'a, T> {
    type Lifted = ty::ParamEnvAnd<'tcx, T::Lifted>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.param_env).and_then(|param_env| {
            tcx.lift(&self.value).map(|value| {
                ty::ParamEnvAnd {
                    param_env,
                    value,
                }
            })
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ClosureSubsts<'a> {
    type Lifted = ty::ClosureSubsts<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.substs).map(|substs| {
            ty::ClosureSubsts { substs }
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::GeneratorSubsts<'a> {
    type Lifted = ty::GeneratorSubsts<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.substs).map(|substs| {
            ty::GeneratorSubsts { substs }
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::adjustment::Adjustment<'a> {
    type Lifted = ty::adjustment::Adjustment<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.kind).and_then(|kind| {
            tcx.lift(&self.target).map(|target| {
                ty::adjustment::Adjustment { kind, target }
            })
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::adjustment::Adjust<'a> {
    type Lifted = ty::adjustment::Adjust<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        match *self {
            ty::adjustment::Adjust::NeverToAny =>
                Some(ty::adjustment::Adjust::NeverToAny),
            ty::adjustment::Adjust::Pointer(ptr) =>
                Some(ty::adjustment::Adjust::Pointer(ptr)),
            ty::adjustment::Adjust::Deref(ref overloaded) => {
                tcx.lift(overloaded).map(ty::adjustment::Adjust::Deref)
            }
            ty::adjustment::Adjust::Borrow(ref autoref) => {
                tcx.lift(autoref).map(ty::adjustment::Adjust::Borrow)
            }
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::adjustment::OverloadedDeref<'a> {
    type Lifted = ty::adjustment::OverloadedDeref<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.region).map(|region| {
            ty::adjustment::OverloadedDeref {
                region,
                mutbl: self.mutbl,
            }
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::adjustment::AutoBorrow<'a> {
    type Lifted = ty::adjustment::AutoBorrow<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        match *self {
            ty::adjustment::AutoBorrow::Ref(r, m) => {
                tcx.lift(&r).map(|r| ty::adjustment::AutoBorrow::Ref(r, m))
            }
            ty::adjustment::AutoBorrow::RawPtr(m) => {
                Some(ty::adjustment::AutoBorrow::RawPtr(m))
            }
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::GenSig<'a> {
    type Lifted = ty::GenSig<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&(self.yield_ty, self.return_ty))
           .map(|(yield_ty, return_ty)| {
               ty::GenSig {
                   yield_ty,
                   return_ty,
               }
           })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::FnSig<'a> {
    type Lifted = ty::FnSig<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.inputs_and_output).map(|x| {
            ty::FnSig {
                inputs_and_output: x,
                c_variadic: self.c_variadic,
                unsafety: self.unsafety,
                abi: self.abi,
            }
        })
    }
}

impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for ty::error::ExpectedFound<T> {
    type Lifted = ty::error::ExpectedFound<T::Lifted>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.expected).and_then(|expected| {
            tcx.lift(&self.found).map(|found| {
                ty::error::ExpectedFound {
                    expected,
                    found,
                }
            })
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::error::TypeError<'a> {
    type Lifted = ty::error::TypeError<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        use crate::ty::error::TypeError::*;

        Some(match *self {
            Mismatch => Mismatch,
            UnsafetyMismatch(x) => UnsafetyMismatch(x),
            AbiMismatch(x) => AbiMismatch(x),
            Mutability => Mutability,
            TupleSize(x) => TupleSize(x),
            FixedArraySize(x) => FixedArraySize(x),
            ArgCount => ArgCount,
            RegionsDoesNotOutlive(a, b) => {
                return tcx.lift(&(a, b)).map(|(a, b)| RegionsDoesNotOutlive(a, b))
            }
            RegionsInsufficientlyPolymorphic(a, b) => {
                return tcx.lift(&b).map(|b| RegionsInsufficientlyPolymorphic(a, b))
            }
            RegionsOverlyPolymorphic(a, b) => {
                return tcx.lift(&b).map(|b| RegionsOverlyPolymorphic(a, b))
            }
            RegionsPlaceholderMismatch => RegionsPlaceholderMismatch,
            IntMismatch(x) => IntMismatch(x),
            FloatMismatch(x) => FloatMismatch(x),
            Traits(x) => Traits(x),
            VariadicMismatch(x) => VariadicMismatch(x),
            CyclicTy(t) => return tcx.lift(&t).map(|t| CyclicTy(t)),
            ProjectionMismatched(x) => ProjectionMismatched(x),
            ProjectionBoundsLength(x) => ProjectionBoundsLength(x),
            Sorts(ref x) => return tcx.lift(x).map(Sorts),
            ExistentialMismatch(ref x) => return tcx.lift(x).map(ExistentialMismatch),
            ConstMismatch(ref x) => return tcx.lift(x).map(ConstMismatch),
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::InstanceDef<'a> {
    type Lifted = ty::InstanceDef<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        match *self {
            ty::InstanceDef::Item(def_id) =>
                Some(ty::InstanceDef::Item(def_id)),
            ty::InstanceDef::VtableShim(def_id) =>
                Some(ty::InstanceDef::VtableShim(def_id)),
            ty::InstanceDef::Intrinsic(def_id) =>
                Some(ty::InstanceDef::Intrinsic(def_id)),
            ty::InstanceDef::FnPtrShim(def_id, ref ty) =>
                Some(ty::InstanceDef::FnPtrShim(def_id, tcx.lift(ty)?)),
            ty::InstanceDef::Virtual(def_id, n) =>
                Some(ty::InstanceDef::Virtual(def_id, n)),
            ty::InstanceDef::ClosureOnceShim { call_once } =>
                Some(ty::InstanceDef::ClosureOnceShim { call_once }),
            ty::InstanceDef::DropGlue(def_id, ref ty) =>
                Some(ty::InstanceDef::DropGlue(def_id, tcx.lift(ty)?)),
            ty::InstanceDef::CloneShim(def_id, ref ty) =>
                Some(ty::InstanceDef::CloneShim(def_id, tcx.lift(ty)?)),
        }
    }
}

BraceStructLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for ty::TypeAndMut<'a> {
        type Lifted = ty::TypeAndMut<'tcx>;
        ty, mutbl
    }
}

BraceStructLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for ty::Instance<'a> {
        type Lifted = ty::Instance<'tcx>;
        def, substs
    }
}

BraceStructLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for interpret::GlobalId<'a> {
        type Lifted = interpret::GlobalId<'tcx>;
        instance, promoted
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
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, _folder: &mut F) -> Self {
        *self
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, _visitor: &mut V) -> bool {
        false
    }
}

impl<'tcx, T: TypeFoldable<'tcx>, U: TypeFoldable<'tcx>> TypeFoldable<'tcx> for (T, U) {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> (T, U) {
        (self.0.fold_with(folder), self.1.fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.0.visit_with(visitor) || self.1.visit_with(visitor)
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
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        Rc::new((**self).fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        (**self).visit_with(visitor)
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Arc<T> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        Arc::new((**self).fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        (**self).visit_with(visitor)
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Box<T> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        let content: T = (**self).fold_with(folder);
        box content
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        (**self).visit_with(visitor)
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Vec<T> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        self.iter().map(|t| t.fold_with(folder)).collect()
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|t| t.visit_with(visitor))
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Box<[T]> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        self.iter().map(|t| t.fold_with(folder)).collect::<Vec<_>>().into_boxed_slice()
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|t| t.visit_with(visitor))
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for ty::Binder<T> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        self.map_bound_ref(|ty| ty.fold_with(folder))
    }

    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_binder(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.skip_binder().visit_with(visitor)
    }

    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        visitor.visit_binder(self)
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::ParamEnv<'tcx> { reveal, caller_bounds, def_id }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::List<ty::ExistentialPredicate<'tcx>> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        let v = self.iter().map(|p| p.fold_with(folder)).collect::<SmallVec<[_; 8]>>();
        folder.tcx().intern_existential_predicates(&v)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|p| p.visit_with(visitor))
    }
}

EnumTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::ExistentialPredicate<'tcx> {
        (ty::ExistentialPredicate::Trait)(a),
        (ty::ExistentialPredicate::Projection)(a),
        (ty::ExistentialPredicate::AutoTrait)(a),
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::List<Ty<'tcx>> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        let v = self.iter().map(|t| t.fold_with(folder)).collect::<SmallVec<[_; 8]>>();
        folder.tcx().intern_type_list(&v)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|t| t.visit_with(visitor))
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::List<ProjectionKind> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        let v = self.iter().map(|t| t.fold_with(folder)).collect::<SmallVec<[_; 8]>>();
        folder.tcx().intern_projs(&v)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|t| t.visit_with(visitor))
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::instance::Instance<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        use crate::ty::InstanceDef::*;
        Self {
            substs: self.substs.fold_with(folder),
            def: match self.def {
                Item(did) => Item(did.fold_with(folder)),
                VtableShim(did) => VtableShim(did.fold_with(folder)),
                Intrinsic(did) => Intrinsic(did.fold_with(folder)),
                FnPtrShim(did, ty) => FnPtrShim(
                    did.fold_with(folder),
                    ty.fold_with(folder),
                ),
                Virtual(did, i) => Virtual(
                    did.fold_with(folder),
                    i,
                ),
                ClosureOnceShim { call_once } => ClosureOnceShim {
                    call_once: call_once.fold_with(folder),
                },
                DropGlue(did, ty) => DropGlue(
                    did.fold_with(folder),
                    ty.fold_with(folder),
                ),
                CloneShim(did, ty) => CloneShim(
                    did.fold_with(folder),
                    ty.fold_with(folder),
                ),
            },
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        use crate::ty::InstanceDef::*;
        self.substs.visit_with(visitor) ||
        match self.def {
            Item(did) | VtableShim(did) | Intrinsic(did) | Virtual(did, _) => {
                did.visit_with(visitor)
            },
            FnPtrShim(did, ty) | CloneShim(did, ty) => {
                did.visit_with(visitor) || ty.visit_with(visitor)
            },
            DropGlue(did, ty) => {
                did.visit_with(visitor) || ty.visit_with(visitor)
            },
            ClosureOnceShim { call_once } => call_once.visit_with(visitor),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for interpret::GlobalId<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        Self {
            instance: self.instance.fold_with(folder),
            promoted: self.promoted
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.instance.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for Ty<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        let sty = match self.sty {
            ty::RawPtr(tm) => ty::RawPtr(tm.fold_with(folder)),
            ty::Array(typ, sz) => ty::Array(typ.fold_with(folder), sz.fold_with(folder)),
            ty::Slice(typ) => ty::Slice(typ.fold_with(folder)),
            ty::Adt(tid, substs) => ty::Adt(tid, substs.fold_with(folder)),
            ty::Dynamic(ref trait_ty, ref region) =>
                ty::Dynamic(trait_ty.fold_with(folder), region.fold_with(folder)),
            ty::Tuple(ts) => ty::Tuple(ts.fold_with(folder)),
            ty::FnDef(def_id, substs) => {
                ty::FnDef(def_id, substs.fold_with(folder))
            }
            ty::FnPtr(f) => ty::FnPtr(f.fold_with(folder)),
            ty::Ref(ref r, ty, mutbl) => {
                ty::Ref(r.fold_with(folder), ty.fold_with(folder), mutbl)
            }
            ty::Generator(did, substs, movability) => {
                ty::Generator(
                    did,
                    substs.fold_with(folder),
                    movability)
            }
            ty::GeneratorWitness(types) => ty::GeneratorWitness(types.fold_with(folder)),
            ty::Closure(did, substs) => ty::Closure(did, substs.fold_with(folder)),
            ty::Projection(ref data) => ty::Projection(data.fold_with(folder)),
            ty::UnnormalizedProjection(ref data) => {
                ty::UnnormalizedProjection(data.fold_with(folder))
            }
            ty::Opaque(did, substs) => ty::Opaque(did, substs.fold_with(folder)),

            ty::Bool |
            ty::Char |
            ty::Str |
            ty::Int(_) |
            ty::Uint(_) |
            ty::Float(_) |
            ty::Error |
            ty::Infer(_) |
            ty::Param(..) |
            ty::Bound(..) |
            ty::Placeholder(..) |
            ty::Never |
            ty::Foreign(..) => return self
        };

        if self.sty == sty {
            self
        } else {
            folder.tcx().mk_ty(sty)
        }
    }

    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_ty(*self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        match self.sty {
            ty::RawPtr(ref tm) => tm.visit_with(visitor),
            ty::Array(typ, sz) => typ.visit_with(visitor) || sz.visit_with(visitor),
            ty::Slice(typ) => typ.visit_with(visitor),
            ty::Adt(_, substs) => substs.visit_with(visitor),
            ty::Dynamic(ref trait_ty, ref reg) =>
                trait_ty.visit_with(visitor) || reg.visit_with(visitor),
            ty::Tuple(ts) => ts.visit_with(visitor),
            ty::FnDef(_, substs) => substs.visit_with(visitor),
            ty::FnPtr(ref f) => f.visit_with(visitor),
            ty::Ref(r, ty, _) => r.visit_with(visitor) || ty.visit_with(visitor),
            ty::Generator(_did, ref substs, _) => {
                substs.visit_with(visitor)
            }
            ty::GeneratorWitness(ref types) => types.visit_with(visitor),
            ty::Closure(_did, ref substs) => substs.visit_with(visitor),
            ty::Projection(ref data) | ty::UnnormalizedProjection(ref data) => {
                data.visit_with(visitor)
            }
            ty::Opaque(_, ref substs) => substs.visit_with(visitor),

            ty::Bool |
            ty::Char |
            ty::Str |
            ty::Int(_) |
            ty::Uint(_) |
            ty::Float(_) |
            ty::Error |
            ty::Infer(_) |
            ty::Bound(..) |
            ty::Placeholder(..) |
            ty::Param(..) |
            ty::Never |
            ty::Foreign(..) => false,
        }
    }

    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        visitor.visit_ty(self)
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::TypeAndMut<'tcx> {
        ty, mutbl
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::GenSig<'tcx> {
        yield_ty, return_ty
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::FnSig<'tcx> {
        inputs_and_output, c_variadic, unsafety, abi
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::TraitRef<'tcx> { def_id, substs }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::ExistentialTraitRef<'tcx> { def_id, substs }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::ImplHeader<'tcx> {
        impl_def_id,
        self_ty,
        trait_ref,
        predicates,
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::Region<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, _folder: &mut F) -> Self {
        *self
    }

    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_region(*self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, _visitor: &mut V) -> bool {
        false
    }

    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        visitor.visit_region(*self)
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::ClosureSubsts<'tcx> {
        substs,
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::GeneratorSubsts<'tcx> {
        substs,
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::adjustment::Adjustment<'tcx> {
        kind,
        target,
    }
}

EnumTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::adjustment::Adjust<'tcx> {
        (ty::adjustment::Adjust::NeverToAny),
        (ty::adjustment::Adjust::Pointer)(a),
        (ty::adjustment::Adjust::Deref)(a),
        (ty::adjustment::Adjust::Borrow)(a),
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::adjustment::OverloadedDeref<'tcx> {
        region, mutbl,
    }
}

EnumTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::adjustment::AutoBorrow<'tcx> {
        (ty::adjustment::AutoBorrow::Ref)(a, b),
        (ty::adjustment::AutoBorrow::RawPtr)(m),
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::GenericPredicates<'tcx> {
        parent, predicates
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::List<ty::Predicate<'tcx>> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        let v = self.iter().map(|p| p.fold_with(folder)).collect::<SmallVec<[_; 8]>>();
        folder.tcx().intern_predicates(&v)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|p| p.visit_with(visitor))
    }
}

EnumTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::Predicate<'tcx> {
        (ty::Predicate::Trait)(a),
        (ty::Predicate::Subtype)(a),
        (ty::Predicate::RegionOutlives)(a),
        (ty::Predicate::TypeOutlives)(a),
        (ty::Predicate::Projection)(a),
        (ty::Predicate::WellFormed)(a),
        (ty::Predicate::ClosureKind)(a, b, c),
        (ty::Predicate::ObjectSafe)(a),
        (ty::Predicate::ConstEvaluatable)(a, b),
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::ProjectionPredicate<'tcx> {
        projection_ty, ty
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::ExistentialProjection<'tcx> {
        ty, substs, item_def_id
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::ProjectionTy<'tcx> {
        substs, item_def_id
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::InstantiatedPredicates<'tcx> {
        predicates
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx, T> TypeFoldable<'tcx> for ty::ParamEnvAnd<'tcx, T> {
        param_env, value
    } where T: TypeFoldable<'tcx>
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::SubtypePredicate<'tcx> {
        a_is_expected, a, b
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::TraitPredicate<'tcx> {
        trait_ref
    }
}

TupleStructTypeFoldableImpl! {
    impl<'tcx,T,U> TypeFoldable<'tcx> for ty::OutlivesPredicate<T,U> {
        a, b
    } where T : TypeFoldable<'tcx>, U : TypeFoldable<'tcx>,
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::ClosureUpvar<'tcx> {
        res, span, ty
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx, T> TypeFoldable<'tcx> for ty::error::ExpectedFound<T> {
        expected, found
    } where T: TypeFoldable<'tcx>
}

impl<'tcx, T: TypeFoldable<'tcx>, I: Idx> TypeFoldable<'tcx> for IndexVec<I, T> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        self.iter().map(|x| x.fold_with(folder)).collect()
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|t| t.visit_with(visitor))
    }
}

EnumTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ty::error::TypeError<'tcx> {
        (ty::error::TypeError::Mismatch),
        (ty::error::TypeError::UnsafetyMismatch)(x),
        (ty::error::TypeError::AbiMismatch)(x),
        (ty::error::TypeError::Mutability),
        (ty::error::TypeError::TupleSize)(x),
        (ty::error::TypeError::FixedArraySize)(x),
        (ty::error::TypeError::ArgCount),
        (ty::error::TypeError::RegionsDoesNotOutlive)(a, b),
        (ty::error::TypeError::RegionsInsufficientlyPolymorphic)(a, b),
        (ty::error::TypeError::RegionsOverlyPolymorphic)(a, b),
        (ty::error::TypeError::RegionsPlaceholderMismatch),
        (ty::error::TypeError::IntMismatch)(x),
        (ty::error::TypeError::FloatMismatch)(x),
        (ty::error::TypeError::Traits)(x),
        (ty::error::TypeError::VariadicMismatch)(x),
        (ty::error::TypeError::CyclicTy)(t),
        (ty::error::TypeError::ProjectionMismatched)(x),
        (ty::error::TypeError::ProjectionBoundsLength)(x),
        (ty::error::TypeError::Sorts)(x),
        (ty::error::TypeError::ExistentialMismatch)(x),
        (ty::error::TypeError::ConstMismatch)(x),
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::Const<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        let ty = self.ty.fold_with(folder);
        let val = self.val.fold_with(folder);
        folder.tcx().mk_const(ty::Const {
            ty,
            val
        })
    }

    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_const(*self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.ty.visit_with(visitor) || self.val.visit_with(visitor)
    }

    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        visitor.visit_const(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ConstValue<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        match *self {
            ConstValue::ByRef { offset, align, alloc } =>
                ConstValue::ByRef { offset, align, alloc },
            ConstValue::Infer(ic) => ConstValue::Infer(ic.fold_with(folder)),
            ConstValue::Param(p) => ConstValue::Param(p.fold_with(folder)),
            ConstValue::Placeholder(p) => ConstValue::Placeholder(p),
            ConstValue::Scalar(a) => ConstValue::Scalar(a),
            ConstValue::Slice { data, start, end } => ConstValue::Slice { data, start, end },
            ConstValue::Unevaluated(did, substs)
                => ConstValue::Unevaluated(did, substs.fold_with(folder)),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        match *self {
            ConstValue::ByRef { .. } => false,
            ConstValue::Infer(ic) => ic.visit_with(visitor),
            ConstValue::Param(p) => p.visit_with(visitor),
            ConstValue::Placeholder(_) => false,
            ConstValue::Scalar(_) => false,
            ConstValue::Slice { .. } => false,
            ConstValue::Unevaluated(_, substs) => substs.visit_with(visitor),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for InferConst<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, _folder: &mut F) -> Self {
        *self
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, _visitor: &mut V) -> bool {
        false
    }
}
