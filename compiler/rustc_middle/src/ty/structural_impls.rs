//! This module contains implementations of the `Lift`, `TypeFoldable` and
//! `TypeVisitable` traits for various types in the Rust compiler. Most are
//! written by hand, though we've recently added some macros and proc-macros
//! to help with the tedium.

use crate::mir::interpret;
use crate::ty::fold::{FallibleTypeFolder, TypeFoldable, TypeSuperFoldable};
use crate::ty::print::{with_no_trimmed_paths, FmtPrinter, Printer};
use crate::ty::visit::{TypeSuperVisitable, TypeVisitable, TypeVisitor};
use crate::ty::{self, AliasTy, InferConst, Lift, Term, TermKind, Ty, TyCtxt};
use rustc_hir::def::Namespace;
use rustc_index::{Idx, IndexVec};
use rustc_target::abi::TyAndLayout;
use rustc_type_ir::{ConstKind, DebugWithInfcx, InferCtxtLike, OptWithInfcx};

use std::fmt::{self, Debug};
use std::ops::ControlFlow;
use std::rc::Rc;
use std::sync::Arc;

use super::{GenericArg, GenericArgKind, Region};

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
            ty::BrAnon(span) => write!(f, "BrAnon({span:?})"),
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
        OptWithInfcx::new_no_ctx(self).fmt(f)
    }
}
impl<'tcx> DebugWithInfcx<TyCtxt<'tcx>> for ty::FnSig<'tcx> {
    fn fmt<InfCtx: InferCtxtLike<TyCtxt<'tcx>>>(
        this: OptWithInfcx<'_, TyCtxt<'tcx>, InfCtx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        let sig = this.data;
        let ty::FnSig { inputs_and_output: _, c_variadic, unsafety, abi } = sig;

        write!(f, "{}", unsafety.prefix_str())?;
        match abi {
            rustc_target::spec::abi::Abi::Rust => (),
            abi => write!(f, "extern \"{abi:?}\" ")?,
        };

        write!(f, "fn(")?;
        let inputs = sig.inputs();
        match inputs.len() {
            0 if *c_variadic => write!(f, "...)")?,
            0 => write!(f, ")")?,
            _ => {
                for ty in &sig.inputs()[0..(sig.inputs().len() - 1)] {
                    write!(f, "{:?}, ", &this.wrap(ty))?;
                }
                write!(f, "{:?}", &this.wrap(sig.inputs().last().unwrap()))?;
                if *c_variadic {
                    write!(f, "...")?;
                }
                write!(f, ")")?;
            }
        }

        match sig.output().kind() {
            ty::Tuple(list) if list.is_empty() => Ok(()),
            _ => write!(f, " -> {:?}", &this.wrap(sig.output())),
        }
    }
}

impl<'tcx> fmt::Debug for ty::ConstVid<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "?{}c", self.index)
    }
}

impl<'tcx> fmt::Debug for ty::TraitRef<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        with_no_trimmed_paths!(fmt::Display::fmt(self, f))
    }
}

impl<'tcx> ty::DebugWithInfcx<TyCtxt<'tcx>> for Ty<'tcx> {
    fn fmt<InfCtx: InferCtxtLike<TyCtxt<'tcx>>>(
        this: OptWithInfcx<'_, TyCtxt<'tcx>, InfCtx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        this.data.fmt(f)
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
        write!(f, "{:?}", self.kind())
    }
}

impl<'tcx> fmt::Debug for ty::ClauseKind<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ty::ClauseKind::ConstArgHasType(ct, ty) => write!(f, "ConstArgHasType({ct:?}, {ty:?})"),
            ty::ClauseKind::Trait(ref a) => a.fmt(f),
            ty::ClauseKind::RegionOutlives(ref pair) => pair.fmt(f),
            ty::ClauseKind::TypeOutlives(ref pair) => pair.fmt(f),
            ty::ClauseKind::Projection(ref pair) => pair.fmt(f),
            ty::ClauseKind::WellFormed(ref data) => write!(f, "WellFormed({:?})", data),
            ty::ClauseKind::ConstEvaluatable(ct) => {
                write!(f, "ConstEvaluatable({ct:?})")
            }
        }
    }
}

impl<'tcx> fmt::Debug for ty::PredicateKind<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ty::PredicateKind::Clause(ref a) => a.fmt(f),
            ty::PredicateKind::Subtype(ref pair) => pair.fmt(f),
            ty::PredicateKind::Coerce(ref pair) => pair.fmt(f),
            ty::PredicateKind::ObjectSafe(trait_def_id) => {
                write!(f, "ObjectSafe({:?})", trait_def_id)
            }
            ty::PredicateKind::ClosureKind(closure_def_id, closure_args, kind) => {
                write!(f, "ClosureKind({:?}, {:?}, {:?})", closure_def_id, closure_args, kind)
            }
            ty::PredicateKind::ConstEquate(c1, c2) => write!(f, "ConstEquate({:?}, {:?})", c1, c2),
            ty::PredicateKind::Ambiguous => write!(f, "Ambiguous"),
            ty::PredicateKind::AliasRelate(t1, t2, dir) => {
                write!(f, "AliasRelate({t1:?}, {dir:?}, {t2:?})")
            }
        }
    }
}

impl<'tcx> fmt::Debug for AliasTy<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        OptWithInfcx::new_no_ctx(self).fmt(f)
    }
}
impl<'tcx> DebugWithInfcx<TyCtxt<'tcx>> for AliasTy<'tcx> {
    fn fmt<InfCtx: InferCtxtLike<TyCtxt<'tcx>>>(
        this: OptWithInfcx<'_, TyCtxt<'tcx>, InfCtx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        f.debug_struct("AliasTy")
            .field("args", &this.map(|data| data.args))
            .field("def_id", &this.data.def_id)
            .finish()
    }
}

impl<'tcx> fmt::Debug for ty::InferConst<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferConst::Var(var) => write!(f, "{var:?}"),
            InferConst::Fresh(var) => write!(f, "Fresh({var:?})"),
        }
    }
}
impl<'tcx> DebugWithInfcx<TyCtxt<'tcx>> for ty::InferConst<'tcx> {
    fn fmt<InfCtx: InferCtxtLike<TyCtxt<'tcx>>>(
        this: OptWithInfcx<'_, TyCtxt<'tcx>, InfCtx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        use ty::InferConst::*;
        match this.infcx.and_then(|infcx| infcx.universe_of_ct(*this.data)) {
            None => write!(f, "{:?}", this.data),
            Some(universe) => match *this.data {
                Var(vid) => write!(f, "?{}_{}c", vid.index, universe.index()),
                Fresh(_) => {
                    unreachable!()
                }
            },
        }
    }
}

impl<'tcx> fmt::Debug for ty::consts::Expr<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        OptWithInfcx::new_no_ctx(self).fmt(f)
    }
}
impl<'tcx> DebugWithInfcx<TyCtxt<'tcx>> for ty::consts::Expr<'tcx> {
    fn fmt<InfCtx: InferCtxtLike<TyCtxt<'tcx>>>(
        this: OptWithInfcx<'_, TyCtxt<'tcx>, InfCtx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        match this.data {
            ty::Expr::Binop(op, lhs, rhs) => {
                write!(f, "({op:?}: {:?}, {:?})", &this.wrap(lhs), &this.wrap(rhs))
            }
            ty::Expr::UnOp(op, rhs) => write!(f, "({op:?}: {:?})", &this.wrap(rhs)),
            ty::Expr::FunctionCall(func, args) => {
                write!(f, "{:?}(", &this.wrap(func))?;
                for arg in args.as_slice().iter().rev().skip(1).rev() {
                    write!(f, "{:?}, ", &this.wrap(arg))?;
                }
                if let Some(arg) = args.last() {
                    write!(f, "{:?}", &this.wrap(arg))?;
                }

                write!(f, ")")
            }
            ty::Expr::Cast(cast_kind, lhs, rhs) => {
                write!(f, "({cast_kind:?}: {:?}, {:?})", &this.wrap(lhs), &this.wrap(rhs))
            }
        }
    }
}

impl<'tcx> fmt::Debug for ty::UnevaluatedConst<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        OptWithInfcx::new_no_ctx(self).fmt(f)
    }
}
impl<'tcx> DebugWithInfcx<TyCtxt<'tcx>> for ty::UnevaluatedConst<'tcx> {
    fn fmt<InfCtx: InferCtxtLike<TyCtxt<'tcx>>>(
        this: OptWithInfcx<'_, TyCtxt<'tcx>, InfCtx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        f.debug_struct("UnevaluatedConst")
            .field("def", &this.data.def)
            .field("args", &this.wrap(this.data.args))
            .finish()
    }
}

impl<'tcx> fmt::Debug for ty::Const<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        OptWithInfcx::new_no_ctx(self).fmt(f)
    }
}
impl<'tcx> DebugWithInfcx<TyCtxt<'tcx>> for ty::Const<'tcx> {
    fn fmt<InfCtx: InferCtxtLike<TyCtxt<'tcx>>>(
        this: OptWithInfcx<'_, TyCtxt<'tcx>, InfCtx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        // This reflects what `Const` looked liked before `Interned` was
        // introduced. We print it like this to avoid having to update expected
        // output in a lot of tests.
        write!(
            f,
            "Const {{ ty: {:?}, kind: {:?} }}",
            &this.map(|data| data.ty()),
            &this.map(|data| data.kind())
        )
    }
}

impl fmt::Debug for ty::BoundTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            ty::BoundTyKind::Anon => write!(f, "{:?}", self.var),
            ty::BoundTyKind::Param(_, sym) => write!(f, "{sym:?}"),
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for ty::Placeholder<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.universe == ty::UniverseIndex::ROOT {
            write!(f, "!{:?}", self.bound)
        } else {
            write!(f, "!{}_{:?}", self.universe.index(), self.bound)
        }
    }
}

impl<'tcx> fmt::Debug for GenericArg<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.unpack() {
            GenericArgKind::Lifetime(lt) => lt.fmt(f),
            GenericArgKind::Type(ty) => ty.fmt(f),
            GenericArgKind::Const(ct) => ct.fmt(f),
        }
    }
}
impl<'tcx> DebugWithInfcx<TyCtxt<'tcx>> for GenericArg<'tcx> {
    fn fmt<InfCtx: InferCtxtLike<TyCtxt<'tcx>>>(
        this: OptWithInfcx<'_, TyCtxt<'tcx>, InfCtx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        match this.data.unpack() {
            GenericArgKind::Lifetime(lt) => write!(f, "{:?}", &this.wrap(lt)),
            GenericArgKind::Const(ct) => write!(f, "{:?}", &this.wrap(ct)),
            GenericArgKind::Type(ty) => write!(f, "{:?}", &this.wrap(ty)),
        }
    }
}

impl<'tcx> fmt::Debug for Region<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.kind())
    }
}
impl<'tcx> DebugWithInfcx<TyCtxt<'tcx>> for Region<'tcx> {
    fn fmt<InfCtx: InferCtxtLike<TyCtxt<'tcx>>>(
        this: OptWithInfcx<'_, TyCtxt<'tcx>, InfCtx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{:?}", &this.map(|data| data.kind()))
    }
}

impl<'tcx> DebugWithInfcx<TyCtxt<'tcx>> for ty::RegionVid {
    fn fmt<InfCtx: InferCtxtLike<TyCtxt<'tcx>>>(
        this: OptWithInfcx<'_, TyCtxt<'tcx>, InfCtx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        match this.infcx.and_then(|infcx| infcx.universe_of_lt(*this.data)) {
            Some(universe) => write!(f, "'?{}_{}", this.data.index(), universe.index()),
            None => write!(f, "{:?}", this.data),
        }
    }
}

impl<'tcx, T: DebugWithInfcx<TyCtxt<'tcx>>> DebugWithInfcx<TyCtxt<'tcx>> for ty::Binder<'tcx, T> {
    fn fmt<InfCtx: InferCtxtLike<TyCtxt<'tcx>>>(
        this: OptWithInfcx<'_, TyCtxt<'tcx>, InfCtx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        f.debug_tuple("Binder")
            .field(&this.map(|data| data.as_ref().skip_binder()))
            .field(&this.data.bound_vars())
            .finish()
    }
}

///////////////////////////////////////////////////////////////////////////
// Atomic structs
//
// For things that don't carry any arena-allocated data (and are
// copy...), just add them to one of these lists as appropriate.

// For things for which the type library provides traversal implementations
// for all Interners, we only need to provide a Lift implementation:
CloneLiftImpls! {
    (),
    bool,
    usize,
    u8,
    u16,
    u32,
    u64,
    String,
    rustc_type_ir::DebruijnIndex,
}

// For things about which the type library does not know, or does not
// provide any traversal implementations, we need to provide both a Lift
// implementation and traversal implementations (the latter only for
// TyCtxt<'_> interners).
TrivialTypeTraversalAndLiftImpls! {
    ::rustc_target::abi::FieldIdx,
    ::rustc_target::abi::VariantIdx,
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
    crate::ty::AliasRelationDirection,
    crate::ty::Placeholder<crate::ty::BoundRegion>,
    crate::ty::Placeholder<crate::ty::BoundTy>,
    crate::ty::Placeholder<ty::BoundVar>,
    crate::ty::ClosureKind,
    crate::ty::FreeRegion,
    crate::ty::InferTy,
    crate::ty::IntVarValue,
    crate::ty::ParamConst,
    crate::ty::ParamTy,
    crate::ty::adjustment::PointerCoercion,
    crate::ty::RegionVid,
    crate::ty::UniverseIndex,
    crate::ty::Variance,
    ::rustc_span::Span,
    ::rustc_span::symbol::Ident,
    ::rustc_errors::ErrorGuaranteed,
    interpret::Scalar,
    rustc_target::abi::Size,
    ty::BoundVar,
}

TrivialTypeTraversalAndLiftImpls! {
    ty::ValTree<'tcx>,
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
// Traversal implementations.

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for ty::AdtDef<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(
        &self,
        _visitor: &mut V,
    ) -> ControlFlow<V::BreakTy> {
        ControlFlow::Continue(())
    }
}

impl<'tcx, T: TypeFoldable<TyCtxt<'tcx>>> TypeFoldable<TyCtxt<'tcx>> for ty::Binder<'tcx, T> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        folder.try_fold_binder(self)
    }
}

impl<'tcx, T: TypeVisitable<TyCtxt<'tcx>>> TypeVisitable<TyCtxt<'tcx>> for ty::Binder<'tcx, T> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        visitor.visit_binder(self)
    }
}

impl<'tcx, T: TypeFoldable<TyCtxt<'tcx>>> TypeSuperFoldable<TyCtxt<'tcx>> for ty::Binder<'tcx, T> {
    fn try_super_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        self.try_map_bound(|ty| ty.try_fold_with(folder))
    }
}

impl<'tcx, T: TypeVisitable<TyCtxt<'tcx>>> TypeSuperVisitable<TyCtxt<'tcx>>
    for ty::Binder<'tcx, T>
{
    fn super_visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(
        &self,
        visitor: &mut V,
    ) -> ControlFlow<V::BreakTy> {
        self.as_ref().skip_binder().visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        ty::util::fold_list(self, folder, |tcx, v| tcx.mk_poly_existential_predicates(v))
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for &'tcx ty::List<ty::Const<'tcx>> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        ty::util::fold_list(self, folder, |tcx, v| tcx.mk_const_list(v))
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for Ty<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        folder.try_fold_ty(self)
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for Ty<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        visitor.visit_ty(*self)
    }
}

impl<'tcx> TypeSuperFoldable<TyCtxt<'tcx>> for Ty<'tcx> {
    fn try_super_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        let kind = match *self.kind() {
            ty::RawPtr(tm) => ty::RawPtr(tm.try_fold_with(folder)?),
            ty::Array(typ, sz) => ty::Array(typ.try_fold_with(folder)?, sz.try_fold_with(folder)?),
            ty::Slice(typ) => ty::Slice(typ.try_fold_with(folder)?),
            ty::Adt(tid, args) => ty::Adt(tid, args.try_fold_with(folder)?),
            ty::Dynamic(trait_ty, region, representation) => ty::Dynamic(
                trait_ty.try_fold_with(folder)?,
                region.try_fold_with(folder)?,
                representation,
            ),
            ty::Tuple(ts) => ty::Tuple(ts.try_fold_with(folder)?),
            ty::FnDef(def_id, args) => ty::FnDef(def_id, args.try_fold_with(folder)?),
            ty::FnPtr(f) => ty::FnPtr(f.try_fold_with(folder)?),
            ty::Ref(r, ty, mutbl) => {
                ty::Ref(r.try_fold_with(folder)?, ty.try_fold_with(folder)?, mutbl)
            }
            ty::Generator(did, args, movability) => {
                ty::Generator(did, args.try_fold_with(folder)?, movability)
            }
            ty::GeneratorWitness(types) => ty::GeneratorWitness(types.try_fold_with(folder)?),
            ty::GeneratorWitnessMIR(did, args) => {
                ty::GeneratorWitnessMIR(did, args.try_fold_with(folder)?)
            }
            ty::Closure(did, args) => ty::Closure(did, args.try_fold_with(folder)?),
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

        Ok(if *self.kind() == kind { self } else { folder.interner().mk_ty_from_kind(kind) })
    }
}

impl<'tcx> TypeSuperVisitable<TyCtxt<'tcx>> for Ty<'tcx> {
    fn super_visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(
        &self,
        visitor: &mut V,
    ) -> ControlFlow<V::BreakTy> {
        match self.kind() {
            ty::RawPtr(ref tm) => tm.visit_with(visitor),
            ty::Array(typ, sz) => {
                typ.visit_with(visitor)?;
                sz.visit_with(visitor)
            }
            ty::Slice(typ) => typ.visit_with(visitor),
            ty::Adt(_, args) => args.visit_with(visitor),
            ty::Dynamic(ref trait_ty, ref reg, _) => {
                trait_ty.visit_with(visitor)?;
                reg.visit_with(visitor)
            }
            ty::Tuple(ts) => ts.visit_with(visitor),
            ty::FnDef(_, args) => args.visit_with(visitor),
            ty::FnPtr(ref f) => f.visit_with(visitor),
            ty::Ref(r, ty, _) => {
                r.visit_with(visitor)?;
                ty.visit_with(visitor)
            }
            ty::Generator(_did, ref args, _) => args.visit_with(visitor),
            ty::GeneratorWitness(ref types) => types.visit_with(visitor),
            ty::GeneratorWitnessMIR(_did, ref args) => args.visit_with(visitor),
            ty::Closure(_did, ref args) => args.visit_with(visitor),
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

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for ty::Region<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        folder.try_fold_region(self)
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for ty::Region<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        visitor.visit_region(*self)
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for ty::Predicate<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        folder.try_fold_predicate(self)
    }
}

// FIXME(clause): This is wonky
impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for ty::Clause<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(folder.try_fold_predicate(self.as_predicate())?.expect_clause())
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for ty::Predicate<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        visitor.visit_predicate(*self)
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for ty::Clause<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        visitor.visit_predicate(self.as_predicate())
    }
}

impl<'tcx> TypeSuperFoldable<TyCtxt<'tcx>> for ty::Predicate<'tcx> {
    fn try_super_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        let new = self.kind().try_fold_with(folder)?;
        Ok(folder.interner().reuse_or_mk_predicate(self, new))
    }
}

impl<'tcx> TypeSuperVisitable<TyCtxt<'tcx>> for ty::Predicate<'tcx> {
    fn super_visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(
        &self,
        visitor: &mut V,
    ) -> ControlFlow<V::BreakTy> {
        self.kind().visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for &'tcx ty::List<ty::Clause<'tcx>> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        ty::util::fold_list(self, folder, |tcx, v| tcx.mk_clauses(v))
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for ty::Const<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        folder.try_fold_const(self)
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for ty::Const<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        visitor.visit_const(*self)
    }
}

impl<'tcx> TypeSuperFoldable<TyCtxt<'tcx>> for ty::Const<'tcx> {
    fn try_super_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        let ty = self.ty().try_fold_with(folder)?;
        let kind = match self.kind() {
            ConstKind::Param(p) => ConstKind::Param(p.try_fold_with(folder)?),
            ConstKind::Infer(i) => ConstKind::Infer(i.try_fold_with(folder)?),
            ConstKind::Bound(d, b) => {
                ConstKind::Bound(d.try_fold_with(folder)?, b.try_fold_with(folder)?)
            }
            ConstKind::Placeholder(p) => ConstKind::Placeholder(p.try_fold_with(folder)?),
            ConstKind::Unevaluated(uv) => ConstKind::Unevaluated(uv.try_fold_with(folder)?),
            ConstKind::Value(v) => ConstKind::Value(v.try_fold_with(folder)?),
            ConstKind::Error(e) => ConstKind::Error(e.try_fold_with(folder)?),
            ConstKind::Expr(e) => ConstKind::Expr(e.try_fold_with(folder)?),
        };
        if ty != self.ty() || kind != self.kind() {
            Ok(folder.interner().mk_ct_from_kind(kind, ty))
        } else {
            Ok(self)
        }
    }
}

impl<'tcx> TypeSuperVisitable<TyCtxt<'tcx>> for ty::Const<'tcx> {
    fn super_visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(
        &self,
        visitor: &mut V,
    ) -> ControlFlow<V::BreakTy> {
        self.ty().visit_with(visitor)?;
        match self.kind() {
            ConstKind::Param(p) => p.visit_with(visitor),
            ConstKind::Infer(i) => i.visit_with(visitor),
            ConstKind::Bound(d, b) => {
                d.visit_with(visitor)?;
                b.visit_with(visitor)
            }
            ConstKind::Placeholder(p) => p.visit_with(visitor),
            ConstKind::Unevaluated(uv) => uv.visit_with(visitor),
            ConstKind::Value(v) => v.visit_with(visitor),
            ConstKind::Error(e) => e.visit_with(visitor),
            ConstKind::Expr(e) => e.visit_with(visitor),
        }
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for InferConst<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        _folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(self)
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for InferConst<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(
        &self,
        _visitor: &mut V,
    ) -> ControlFlow<V::BreakTy> {
        ControlFlow::Continue(())
    }
}

impl<'tcx> TypeSuperVisitable<TyCtxt<'tcx>> for ty::UnevaluatedConst<'tcx> {
    fn super_visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(
        &self,
        visitor: &mut V,
    ) -> ControlFlow<V::BreakTy> {
        self.args.visit_with(visitor)
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for TyAndLayout<'tcx, Ty<'tcx>> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        visitor.visit_ty(self.ty)
    }
}
