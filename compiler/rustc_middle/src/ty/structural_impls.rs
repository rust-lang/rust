//! This module contains implementations of the `Lift`, `TypeFoldable` and
//! `TypeVisitable` traits for various types in the Rust compiler. Most are
//! written by hand, though we've recently added some macros and proc-macros
//! to help with the tedium.

use std::fmt::{self, Debug};

use rustc_abi::TyAndLayout;
use rustc_hir::def::Namespace;
use rustc_hir::def_id::LocalDefId;
use rustc_span::source_map::Spanned;
use rustc_type_ir::{ConstKind, TypeFolder, VisitorResult, try_visit};

use super::print::PrettyPrinter;
use super::{GenericArg, GenericArgKind, Pattern, Region};
use crate::infer::canonical::CanonicalVarInfos;
use crate::mir::PlaceElem;
use crate::ty::print::{FmtPrinter, Printer, with_no_trimmed_paths};
use crate::ty::{
    self, FallibleTypeFolder, Lift, Term, TermKind, Ty, TyCtxt, TypeFoldable, TypeSuperFoldable,
    TypeSuperVisitable, TypeVisitable, TypeVisitor,
};

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
        let name = ty::tls::with(|tcx| tcx.hir_name(self.var_path.hir_id));
        write!(f, "UpvarId({:?};`{}`;{:?})", self.var_path.hir_id, name, self.closure_expr_id)
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
            ty::BoundRegionKind::Anon => write!(f, "BrAnon"),
            ty::BoundRegionKind::Named(did, name) => {
                if did.is_crate_root() {
                    write!(f, "BrNamed({name})")
                } else {
                    write!(f, "BrNamed({did:?}, {name})")
                }
            }
            ty::BoundRegionKind::ClosureEnv => write!(f, "BrEnv"),
        }
    }
}

impl fmt::Debug for ty::LateParamRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReLateParam({:?}, {:?})", self.scope, self.kind)
    }
}

impl fmt::Debug for ty::LateParamRegionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ty::LateParamRegionKind::Anon(idx) => write!(f, "LateAnon({idx})"),
            ty::LateParamRegionKind::Named(did, name) => {
                if did.is_crate_root() {
                    write!(f, "LateNamed({name})")
                } else {
                    write!(f, "LateNamed({did:?}, {name})")
                }
            }
            ty::LateParamRegionKind::ClosureEnv => write!(f, "LateEnv"),
        }
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

impl<'tcx> fmt::Debug for ty::consts::Expr<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            ty::ExprKind::Binop(op) => {
                let (lhs_ty, rhs_ty, lhs, rhs) = self.binop_args();
                write!(f, "({op:?}: ({:?}: {:?}), ({:?}: {:?}))", lhs, lhs_ty, rhs, rhs_ty,)
            }
            ty::ExprKind::UnOp(op) => {
                let (rhs_ty, rhs) = self.unop_args();
                write!(f, "({op:?}: ({:?}: {:?}))", rhs, rhs_ty)
            }
            ty::ExprKind::FunctionCall => {
                let (func_ty, func, args) = self.call_args();
                let args = args.collect::<Vec<_>>();
                write!(f, "({:?}: {:?})(", func, func_ty)?;
                for arg in args.iter().rev().skip(1).rev() {
                    write!(f, "{:?}, ", arg)?;
                }
                if let Some(arg) = args.last() {
                    write!(f, "{:?}", arg)?;
                }

                write!(f, ")")
            }
            ty::ExprKind::Cast(kind) => {
                let (value_ty, value, to_ty) = self.cast_args();
                write!(f, "({kind:?}: ({:?}: {:?}), {:?})", value, value_ty, to_ty)
            }
        }
    }
}

impl<'tcx> fmt::Debug for ty::Const<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // If this is a value, we spend some effort to make it look nice.
        if let ConstKind::Value(cv) = self.kind() {
            return ty::tls::with(move |tcx| {
                let cv = tcx.lift(cv).unwrap();
                let mut cx = FmtPrinter::new(tcx, Namespace::ValueNS);
                cx.pretty_print_const_valtree(cv, /*print_ty*/ true)?;
                f.write_str(&cx.into_buffer())
            });
        }
        // Fall back to something verbose.
        write!(f, "{:?}", self.kind())
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

impl<'tcx> fmt::Debug for Region<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.kind())
    }
}

///////////////////////////////////////////////////////////////////////////
// Atomic structs
//
// For things that don't carry any arena-allocated data (and are
// copy...), just add them to one of these lists as appropriate.

// For things for which the type library provides traversal implementations
// for all Interners, we only need to provide a Lift implementation.
TrivialLiftImpls! {
    (),
    bool,
    usize,
    u64,
    // tidy-alphabetical-start
    crate::mir::interpret::AllocId,
    crate::mir::interpret::Scalar,
    crate::mir::Promoted,
    rustc_abi::ExternAbi,
    rustc_abi::Size,
    rustc_hir::Safety,
    rustc_type_ir::BoundConstness,
    rustc_type_ir::PredicatePolarity,
    // tidy-alphabetical-end
}

// For some things about which the type library does not know, or does not
// provide any traversal implementations, we need to provide a traversal
// implementation (only for TyCtxt<'_> interners).
TrivialTypeTraversalImpls! {
    // tidy-alphabetical-start
    crate::infer::canonical::Certainty,
    crate::mir::BasicBlock,
    crate::mir::BindingForm<'tcx>,
    crate::mir::BlockTailInfo,
    crate::mir::BorrowKind,
    crate::mir::CastKind,
    crate::mir::ConstValue<'tcx>,
    crate::mir::CoroutineSavedLocal,
    crate::mir::FakeReadCause,
    crate::mir::Local,
    crate::mir::MirPhase,
    crate::mir::NullOp<'tcx>,
    crate::mir::Promoted,
    crate::mir::RawPtrKind,
    crate::mir::RetagKind,
    crate::mir::SourceInfo,
    crate::mir::SourceScope,
    crate::mir::SourceScopeLocalData,
    crate::mir::SwitchTargets,
    crate::traits::IsConstable,
    crate::traits::OverflowError,
    crate::ty::abstract_const::NotConstEvaluatable,
    crate::ty::adjustment::AutoBorrowMutability,
    crate::ty::adjustment::PointerCoercion,
    crate::ty::AdtKind,
    crate::ty::AssocItem,
    crate::ty::AssocKind,
    crate::ty::BoundRegion,
    crate::ty::BoundVar,
    crate::ty::InferConst,
    crate::ty::Placeholder<crate::ty::BoundRegion>,
    crate::ty::Placeholder<crate::ty::BoundTy>,
    crate::ty::Placeholder<ty::BoundVar>,
    crate::ty::UserTypeAnnotationIndex,
    crate::ty::ValTree<'tcx>,
    rustc_abi::FieldIdx,
    rustc_abi::VariantIdx,
    rustc_ast::InlineAsmOptions,
    rustc_ast::InlineAsmTemplatePiece,
    rustc_hir::CoroutineKind,
    rustc_hir::def_id::LocalDefId,
    rustc_hir::HirId,
    rustc_hir::MatchSource,
    rustc_hir::RangeEnd,
    rustc_span::Ident,
    rustc_span::Span,
    rustc_span::Symbol,
    rustc_target::asm::InlineAsmRegOrRegClass,
    // tidy-alphabetical-end
}

// For some things about which the type library does not know, or does not
// provide any traversal implementations, we need to provide a traversal
// implementation and a lift implementation (the former only for TyCtxt<'_>
// interners).
TrivialTypeTraversalAndLiftImpls! {
    // tidy-alphabetical-start
    crate::ty::instance::ReifyReason,
    crate::ty::ParamConst,
    crate::ty::ParamTy,
    rustc_hir::def_id::DefId,
    // tidy-alphabetical-end
}

///////////////////////////////////////////////////////////////////////////
// Lift implementations

impl<'tcx, T: Lift<TyCtxt<'tcx>>> Lift<TyCtxt<'tcx>> for Option<T> {
    type Lifted = Option<T::Lifted>;
    fn lift_to_interner(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        Some(match self {
            Some(x) => Some(tcx.lift(x)?),
            None => None,
        })
    }
}

impl<'a, 'tcx> Lift<TyCtxt<'tcx>> for Term<'a> {
    type Lifted = ty::Term<'tcx>;
    fn lift_to_interner(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        match self.unpack() {
            TermKind::Ty(ty) => tcx.lift(ty).map(Into::into),
            TermKind::Const(c) => tcx.lift(c).map(Into::into),
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Traversal implementations.

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for ty::AdtDef<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, _visitor: &mut V) -> V::Result {
        V::Result::output()
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for Pattern<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        let pat = (*self).clone().try_fold_with(folder)?;
        Ok(if pat == *self { self } else { folder.cx().mk_pat(pat) })
    }

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        let pat = (*self).clone().fold_with(folder);
        if pat == *self { self } else { folder.cx().mk_pat(pat) }
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for Pattern<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        (**self).visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for Ty<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        folder.try_fold_ty(self)
    }

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        folder.fold_ty(self)
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
            ty::RawPtr(ty, mutbl) => ty::RawPtr(ty.try_fold_with(folder)?, mutbl),
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
            ty::FnPtr(sig_tys, hdr) => ty::FnPtr(sig_tys.try_fold_with(folder)?, hdr),
            ty::UnsafeBinder(f) => ty::UnsafeBinder(f.try_fold_with(folder)?),
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
            ty::Pat(ty, pat) => ty::Pat(ty.try_fold_with(folder)?, pat.try_fold_with(folder)?),

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

        Ok(if *self.kind() == kind { self } else { folder.cx().mk_ty_from_kind(kind) })
    }

    fn super_fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        let kind = match *self.kind() {
            ty::RawPtr(ty, mutbl) => ty::RawPtr(ty.fold_with(folder), mutbl),
            ty::Array(typ, sz) => ty::Array(typ.fold_with(folder), sz.fold_with(folder)),
            ty::Slice(typ) => ty::Slice(typ.fold_with(folder)),
            ty::Adt(tid, args) => ty::Adt(tid, args.fold_with(folder)),
            ty::Dynamic(trait_ty, region, representation) => {
                ty::Dynamic(trait_ty.fold_with(folder), region.fold_with(folder), representation)
            }
            ty::Tuple(ts) => ty::Tuple(ts.fold_with(folder)),
            ty::FnDef(def_id, args) => ty::FnDef(def_id, args.fold_with(folder)),
            ty::FnPtr(sig_tys, hdr) => ty::FnPtr(sig_tys.fold_with(folder), hdr),
            ty::UnsafeBinder(f) => ty::UnsafeBinder(f.fold_with(folder)),
            ty::Ref(r, ty, mutbl) => ty::Ref(r.fold_with(folder), ty.fold_with(folder), mutbl),
            ty::Coroutine(did, args) => ty::Coroutine(did, args.fold_with(folder)),
            ty::CoroutineWitness(did, args) => ty::CoroutineWitness(did, args.fold_with(folder)),
            ty::Closure(did, args) => ty::Closure(did, args.fold_with(folder)),
            ty::CoroutineClosure(did, args) => ty::CoroutineClosure(did, args.fold_with(folder)),
            ty::Alias(kind, data) => ty::Alias(kind, data.fold_with(folder)),
            ty::Pat(ty, pat) => ty::Pat(ty.fold_with(folder), pat.fold_with(folder)),

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

        if *self.kind() == kind { self } else { folder.cx().mk_ty_from_kind(kind) }
    }
}

impl<'tcx> TypeSuperVisitable<TyCtxt<'tcx>> for Ty<'tcx> {
    fn super_visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        match self.kind() {
            ty::RawPtr(ty, _mutbl) => ty.visit_with(visitor),
            ty::Array(typ, sz) => {
                try_visit!(typ.visit_with(visitor));
                sz.visit_with(visitor)
            }
            ty::Slice(typ) => typ.visit_with(visitor),
            ty::Adt(_, args) => args.visit_with(visitor),
            ty::Dynamic(trait_ty, reg, _) => {
                try_visit!(trait_ty.visit_with(visitor));
                reg.visit_with(visitor)
            }
            ty::Tuple(ts) => ts.visit_with(visitor),
            ty::FnDef(_, args) => args.visit_with(visitor),
            ty::FnPtr(sig_tys, _) => sig_tys.visit_with(visitor),
            ty::UnsafeBinder(f) => f.visit_with(visitor),
            ty::Ref(r, ty, _) => {
                try_visit!(r.visit_with(visitor));
                ty.visit_with(visitor)
            }
            ty::Coroutine(_did, args) => args.visit_with(visitor),
            ty::CoroutineWitness(_did, args) => args.visit_with(visitor),
            ty::Closure(_did, args) => args.visit_with(visitor),
            ty::CoroutineClosure(_did, args) => args.visit_with(visitor),
            ty::Alias(_, data) => data.visit_with(visitor),

            ty::Pat(ty, pat) => {
                try_visit!(ty.visit_with(visitor));
                pat.visit_with(visitor)
            }

            ty::Error(guar) => guar.visit_with(visitor),

            ty::Bool
            | ty::Char
            | ty::Str
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
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

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        folder.fold_region(self)
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

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        folder.fold_predicate(self)
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

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        folder.fold_predicate(self.as_predicate()).expect_clause()
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
        Ok(folder.cx().reuse_or_mk_predicate(self, new))
    }

    fn super_fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        let new = self.kind().fold_with(folder);
        folder.cx().reuse_or_mk_predicate(self, new)
    }
}

impl<'tcx> TypeSuperVisitable<TyCtxt<'tcx>> for ty::Predicate<'tcx> {
    fn super_visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        self.kind().visit_with(visitor)
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for ty::Clauses<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        visitor.visit_clauses(self)
    }
}

impl<'tcx> TypeSuperVisitable<TyCtxt<'tcx>> for ty::Clauses<'tcx> {
    fn super_visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        self.as_slice().visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for ty::Const<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        folder.try_fold_const(self)
    }

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        folder.fold_const(self)
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
        if kind != self.kind() { Ok(folder.cx().mk_ct_from_kind(kind)) } else { Ok(self) }
    }

    fn super_fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        let kind = match self.kind() {
            ConstKind::Param(p) => ConstKind::Param(p.fold_with(folder)),
            ConstKind::Infer(i) => ConstKind::Infer(i.fold_with(folder)),
            ConstKind::Bound(d, b) => ConstKind::Bound(d.fold_with(folder), b.fold_with(folder)),
            ConstKind::Placeholder(p) => ConstKind::Placeholder(p.fold_with(folder)),
            ConstKind::Unevaluated(uv) => ConstKind::Unevaluated(uv.fold_with(folder)),
            ConstKind::Value(v) => ConstKind::Value(v.fold_with(folder)),
            ConstKind::Error(e) => ConstKind::Error(e.fold_with(folder)),
            ConstKind::Expr(e) => ConstKind::Expr(e.fold_with(folder)),
        };
        if kind != self.kind() { folder.cx().mk_ct_from_kind(kind) } else { self }
    }
}

impl<'tcx> TypeSuperVisitable<TyCtxt<'tcx>> for ty::Const<'tcx> {
    fn super_visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
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

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for rustc_span::ErrorGuaranteed {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        visitor.visit_error(*self)
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for rustc_span::ErrorGuaranteed {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        _folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(self)
    }

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, _folder: &mut F) -> Self {
        self
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

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        Spanned { node: self.node.fold_with(folder), span: self.span.fold_with(folder) }
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for &'tcx ty::List<LocalDefId> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        _folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(self)
    }

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, _folder: &mut F) -> Self {
        self
    }
}

macro_rules! list_fold {
    ($($ty:ty : $mk:ident),+ $(,)?) => {
        $(
            impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for $ty {
                fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
                    self,
                    folder: &mut F,
                ) -> Result<Self, F::Error> {
                    ty::util::try_fold_list(self, folder, |tcx, v| tcx.$mk(v))
                }

                fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(
                    self,
                    folder: &mut F,
                ) -> Self {
                    ty::util::fold_list(self, folder, |tcx, v| tcx.$mk(v))
                }
            }
        )*
    }
}

list_fold! {
    ty::Clauses<'tcx> : mk_clauses,
    &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>> : mk_poly_existential_predicates,
    &'tcx ty::List<PlaceElem<'tcx>> : mk_place_elems,
    CanonicalVarInfos<'tcx> : mk_canonical_var_infos,
}
