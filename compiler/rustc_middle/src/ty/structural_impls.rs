//! This module contains implementations of the `Lift`, `TypeFoldable` and
//! `TypeVisitable` traits for various types in the Rust compiler. Most are
//! written by hand, though we've recently added some macros and proc-macros
//! to help with the tedium.

use crate::mir::interpret;
use crate::ty::fold::{FallibleTypeFolder, TypeFoldable, TypeSuperFoldable};
use crate::ty::print::{with_no_trimmed_paths, FmtPrinter, Printer};
use crate::ty::visit::{TypeSuperVisitable, TypeVisitable, TypeVisitor};
use crate::ty::{self, AliasTy, InferConst, Lift, Term, TermKind, Ty, TyCtxt};
use rustc_ast_ir::try_visit;
use rustc_ast_ir::visit::VisitorResult;
use rustc_hir::def::Namespace;
use rustc_span::source_map::Spanned;
use rustc_target::abi::TyAndLayout;
use rustc_type_ir::{ConstKind, DebugWithInfcx, InferCtxtLike, WithInfcx};

use std::fmt::{self, Debug};

use super::print::PrettyPrinter;
use super::{GenericArg, GenericArgKind, Region};

impl fmt::Debug for ty::TraitDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ty::tls::with(|tcx| {
            with_no_trimmed_paths!({
                let s = FmtPrinter::print_string(tcx, Namespace::TypeNS, |cx| {
                    cx.print_def_path(self.def_id, &[])
                })?;
                f.write_str(&s)
            })
        })
    }
}

impl<'tcx> fmt::Debug for ty::AdtDef<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ty::tls::with(|tcx| {
            with_no_trimmed_paths!({
                let s = FmtPrinter::print_string(tcx, Namespace::TypeNS, |cx| {
                    cx.print_def_path(self.did(), &[])
                })?;
                f.write_str(&s)
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
            ty::BrAnon => write!(f, "BrAnon"),
            ty::BrNamed(did, name) => {
                if did.is_crate_root() {
                    write!(f, "BrNamed({name})")
                } else {
                    write!(f, "BrNamed({did:?}, {name})")
                }
            }
            ty::BrEnv => write!(f, "BrEnv"),
        }
    }
}

impl fmt::Debug for ty::LateParamRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReLateParam({:?}, {:?})", self.scope, self.bound_region)
    }
}

impl<'tcx> fmt::Debug for ty::FnSig<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        WithInfcx::with_no_infcx(self).fmt(f)
    }
}
impl<'tcx> DebugWithInfcx<TyCtxt<'tcx>> for ty::FnSig<'tcx> {
    fn fmt<Infcx: InferCtxtLike<Interner = TyCtxt<'tcx>>>(
        this: WithInfcx<'_, Infcx, &Self>,
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

impl<'tcx> fmt::Debug for ty::TraitRef<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        with_no_trimmed_paths!(fmt::Display::fmt(self, f))
    }
}

impl<'tcx> ty::DebugWithInfcx<TyCtxt<'tcx>> for Ty<'tcx> {
    fn fmt<Infcx: InferCtxtLike<Interner = TyCtxt<'tcx>>>(
        this: WithInfcx<'_, Infcx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        this.data.fmt(f)
    }
}
impl<'tcx> fmt::Debug for Ty<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        with_no_trimmed_paths!(fmt::Debug::fmt(self.kind(), f))
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
        // FIXME(effects) printing?
        write!(f, "TraitPredicate({:?}, polarity:{:?})", self.trait_ref, self.polarity)
    }
}

impl<'tcx> fmt::Debug for ty::ProjectionPredicate<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ProjectionPredicate({:?}, {:?})", self.projection_ty, self.term)
    }
}

impl<'tcx> fmt::Debug for ty::NormalizesTo<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NormalizesTo({:?}, {:?})", self.alias, self.term)
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

impl<'tcx> fmt::Debug for AliasTy<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        WithInfcx::with_no_infcx(self).fmt(f)
    }
}
impl<'tcx> DebugWithInfcx<TyCtxt<'tcx>> for AliasTy<'tcx> {
    fn fmt<Infcx: InferCtxtLike<Interner = TyCtxt<'tcx>>>(
        this: WithInfcx<'_, Infcx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        f.debug_struct("AliasTy")
            .field("args", &this.map(|data| data.args))
            .field("def_id", &this.data.def_id)
            .finish()
    }
}

impl<'tcx> fmt::Debug for ty::consts::Expr<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        WithInfcx::with_no_infcx(self).fmt(f)
    }
}
impl<'tcx> DebugWithInfcx<TyCtxt<'tcx>> for ty::consts::Expr<'tcx> {
    fn fmt<Infcx: InferCtxtLike<Interner = TyCtxt<'tcx>>>(
        this: WithInfcx<'_, Infcx, &Self>,
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
        WithInfcx::with_no_infcx(self).fmt(f)
    }
}
impl<'tcx> DebugWithInfcx<TyCtxt<'tcx>> for ty::UnevaluatedConst<'tcx> {
    fn fmt<Infcx: InferCtxtLike<Interner = TyCtxt<'tcx>>>(
        this: WithInfcx<'_, Infcx, &Self>,
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
        WithInfcx::with_no_infcx(self).fmt(f)
    }
}
impl<'tcx> DebugWithInfcx<TyCtxt<'tcx>> for ty::Const<'tcx> {
    fn fmt<Infcx: InferCtxtLike<Interner = TyCtxt<'tcx>>>(
        this: WithInfcx<'_, Infcx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        // If this is a value, we spend some effort to make it look nice.
        if let ConstKind::Value(_) = this.data.kind() {
            return ty::tls::with(move |tcx| {
                // Somehow trying to lift the valtree results in lifetime errors, so we lift the
                // entire constant.
                let lifted = tcx.lift(*this.data).unwrap();
                let ConstKind::Value(valtree) = lifted.kind() else {
                    bug!("we checked that this is a valtree")
                };
                let mut cx = FmtPrinter::new(tcx, Namespace::ValueNS);
                cx.pretty_print_const_valtree(valtree, lifted.ty(), /*print_ty*/ true)?;
                f.write_str(&cx.into_buffer())
            });
        }
        // Fall back to something verbose.
        write!(
            f,
            "{kind:?}: {ty:?}",
            ty = &this.map(|data| data.ty()),
            kind = &this.map(|data| data.kind())
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
    fn fmt<Infcx: InferCtxtLike<Interner = TyCtxt<'tcx>>>(
        this: WithInfcx<'_, Infcx, &Self>,
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
    fn fmt<Infcx: InferCtxtLike<Interner = TyCtxt<'tcx>>>(
        this: WithInfcx<'_, Infcx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{:?}", &this.map(|data| data.kind()))
    }
}

impl<'tcx> DebugWithInfcx<TyCtxt<'tcx>> for ty::RegionVid {
    fn fmt<Infcx: InferCtxtLike<Interner = TyCtxt<'tcx>>>(
        this: WithInfcx<'_, Infcx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        match this.infcx.universe_of_lt(*this.data) {
            Some(universe) => write!(f, "'?{}_{}", this.data.index(), universe.index()),
            None => write!(f, "{:?}", this.data),
        }
    }
}

impl<'tcx, T: DebugWithInfcx<TyCtxt<'tcx>>> DebugWithInfcx<TyCtxt<'tcx>> for ty::Binder<'tcx, T> {
    fn fmt<Infcx: InferCtxtLike<Interner = TyCtxt<'tcx>>>(
        this: WithInfcx<'_, Infcx, &Self>,
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
TrivialLiftImpls! {
     (),
     bool,
     usize,
     u64,
}

// For some things about which the type library does not know, or does not
// provide any traversal implementations, we need to provide a traversal
// implementation (only for TyCtxt<'_> interners).
TrivialTypeTraversalImpls! {
    ::rustc_target::abi::FieldIdx,
    ::rustc_target::abi::VariantIdx,
    crate::middle::region::Scope,
    crate::ty::FloatTy,
    ::rustc_ast::InlineAsmOptions,
    ::rustc_ast::InlineAsmTemplatePiece,
    ::rustc_ast::NodeId,
    ::rustc_span::symbol::Symbol,
    ::rustc_hir::def::Res,
    ::rustc_hir::def_id::LocalDefId,
    ::rustc_hir::HirId,
    ::rustc_hir::MatchSource,
    ::rustc_target::asm::InlineAsmRegOrRegClass,
    crate::mir::coverage::CounterId,
    crate::mir::coverage::ExpressionId,
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
    crate::ty::Placeholder<crate::ty::BoundRegion>,
    crate::ty::Placeholder<crate::ty::BoundTy>,
    crate::ty::Placeholder<ty::BoundVar>,
    crate::ty::LateParamRegion,
    crate::ty::InferTy,
    crate::ty::IntVarValue,
    crate::ty::adjustment::PointerCoercion,
    crate::ty::RegionVid,
    crate::ty::Variance,
    ::rustc_span::Span,
    ::rustc_span::symbol::Ident,
    ::rustc_errors::ErrorGuaranteed,
    ty::BoundVar,
    ty::ValTree<'tcx>,
}
// For some things about which the type library does not know, or does not
// provide any traversal implementations, we need to provide a traversal
// implementation and a lift implementation (the former only for TyCtxt<'_>
// interners).
TrivialTypeTraversalAndLiftImpls! {
    ::rustc_hir::def_id::DefId,
    ::rustc_hir::Unsafety,
    ::rustc_target::spec::abi::Abi,
    crate::ty::ClosureKind,
    crate::ty::ParamConst,
    crate::ty::ParamTy,
    interpret::AllocId,
    interpret::CtfeProvenance,
    interpret::Scalar,
    rustc_target::abi::Size,
}

///////////////////////////////////////////////////////////////////////////
// Lift implementations

impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for Option<T> {
    type Lifted = Option<T::Lifted>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        Some(match self {
            Some(x) => Some(tcx.lift(x)?),
            None => None,
        })
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

///////////////////////////////////////////////////////////////////////////
// Traversal implementations.

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for ty::AdtDef<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, _visitor: &mut V) -> V::Result {
        V::Result::output()
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
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
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
    fn super_visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
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
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
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
            ty::Coroutine(did, args) => ty::Coroutine(did, args.try_fold_with(folder)?),
            ty::CoroutineWitness(did, args) => {
                ty::CoroutineWitness(did, args.try_fold_with(folder)?)
            }
            ty::Closure(did, args) => ty::Closure(did, args.try_fold_with(folder)?),
            ty::CoroutineClosure(did, args) => {
                ty::CoroutineClosure(did, args.try_fold_with(folder)?)
            }
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
    fn super_visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        match self.kind() {
            ty::RawPtr(ref tm) => tm.visit_with(visitor),
            ty::Array(typ, sz) => {
                try_visit!(typ.visit_with(visitor));
                sz.visit_with(visitor)
            }
            ty::Slice(typ) => typ.visit_with(visitor),
            ty::Adt(_, args) => args.visit_with(visitor),
            ty::Dynamic(ref trait_ty, ref reg, _) => {
                try_visit!(trait_ty.visit_with(visitor));
                reg.visit_with(visitor)
            }
            ty::Tuple(ts) => ts.visit_with(visitor),
            ty::FnDef(_, args) => args.visit_with(visitor),
            ty::FnPtr(ref f) => f.visit_with(visitor),
            ty::Ref(r, ty, _) => {
                try_visit!(r.visit_with(visitor));
                ty.visit_with(visitor)
            }
            ty::Coroutine(_did, ref args) => args.visit_with(visitor),
            ty::CoroutineWitness(_did, ref args) => args.visit_with(visitor),
            ty::Closure(_did, ref args) => args.visit_with(visitor),
            ty::CoroutineClosure(_did, ref args) => args.visit_with(visitor),
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
            | ty::Foreign(..) => V::Result::output(),
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
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
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
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        visitor.visit_predicate(*self)
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for ty::Clause<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
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
    fn super_visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
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
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
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
    fn super_visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        try_visit!(self.ty().visit_with(visitor));
        match self.kind() {
            ConstKind::Param(p) => p.visit_with(visitor),
            ConstKind::Infer(i) => i.visit_with(visitor),
            ConstKind::Bound(d, b) => {
                try_visit!(d.visit_with(visitor));
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

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for InferConst {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        _folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(self)
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for InferConst {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, _visitor: &mut V) -> V::Result {
        V::Result::output()
    }
}

impl<'tcx> TypeSuperVisitable<TyCtxt<'tcx>> for ty::UnevaluatedConst<'tcx> {
    fn super_visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        self.args.visit_with(visitor)
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for TyAndLayout<'tcx, Ty<'tcx>> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        visitor.visit_ty(self.ty)
    }
}

impl<'tcx, T: TypeVisitable<TyCtxt<'tcx>> + Debug + Clone> TypeVisitable<TyCtxt<'tcx>>
    for Spanned<T>
{
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        try_visit!(self.node.visit_with(visitor));
        self.span.visit_with(visitor)
    }
}

impl<'tcx, T: TypeFoldable<TyCtxt<'tcx>> + Debug + Clone> TypeFoldable<TyCtxt<'tcx>>
    for Spanned<T>
{
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(Spanned {
            node: self.node.try_fold_with(folder)?,
            span: self.span.try_fold_with(folder)?,
        })
    }
}
