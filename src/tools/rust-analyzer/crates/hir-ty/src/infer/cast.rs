//! Type cast logic. Basically coercion + additional casts.

use hir_def::{AdtId, hir::ExprId, signatures::TraitFlags};
use rustc_ast_ir::Mutability;
use rustc_type_ir::{
    Flags, InferTy, TypeFlags, UintTy,
    inherent::{AdtDef, BoundExistentialPredicates as _, IntoKind, SliceLike, Ty as _},
};
use stdx::never;

use crate::{
    InferenceDiagnostic,
    db::HirDatabase,
    infer::{AllowTwoPhase, InferenceContext, expr::ExprIsRead},
    next_solver::{BoundExistentialPredicates, DbInterner, ParamTy, Ty, TyKind},
};

#[derive(Debug)]
pub(crate) enum Int {
    I,
    U(UintTy),
    Bool,
    Char,
    CEnum,
    InferenceVar,
}

#[derive(Debug)]
pub(crate) enum CastTy<'db> {
    Int(Int),
    Float,
    FnPtr,
    Ptr(Ty<'db>, Mutability),
    // `DynStar` is Not supported yet in r-a
}

impl<'db> CastTy<'db> {
    pub(crate) fn from_ty(db: &dyn HirDatabase, t: Ty<'db>) -> Option<Self> {
        match t.kind() {
            TyKind::Bool => Some(Self::Int(Int::Bool)),
            TyKind::Char => Some(Self::Int(Int::Char)),
            TyKind::Int(_) => Some(Self::Int(Int::I)),
            TyKind::Uint(it) => Some(Self::Int(Int::U(it))),
            TyKind::Infer(InferTy::IntVar(_)) => Some(Self::Int(Int::InferenceVar)),
            TyKind::Infer(InferTy::FloatVar(_)) => Some(Self::Float),
            TyKind::Float(_) => Some(Self::Float),
            TyKind::Adt(..) => {
                let (AdtId::EnumId(id), _) = t.as_adt()? else {
                    return None;
                };
                let enum_data = id.enum_variants(db);
                if enum_data.is_payload_free(db) { Some(Self::Int(Int::CEnum)) } else { None }
            }
            TyKind::RawPtr(ty, m) => Some(Self::Ptr(ty, m)),
            TyKind::FnPtr(..) => Some(Self::FnPtr),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CastError {
    Unknown,
    CastToBool,
    CastToChar,
    DifferingKinds,
    SizedUnsizedCast,
    IllegalCast,
    IntToFatCast,
    NeedDeref,
    NeedViaPtr,
    NeedViaThinPtr,
    NeedViaInt,
    NonScalar,
    // We don't want to report errors with unknown types currently.
    // UnknownCastPtrKind,
    // UnknownExprPtrKind,
}

impl CastError {
    fn into_diagnostic<'db>(
        self,
        expr: ExprId,
        expr_ty: Ty<'db>,
        cast_ty: Ty<'db>,
    ) -> InferenceDiagnostic<'db> {
        InferenceDiagnostic::InvalidCast { expr, error: self, expr_ty, cast_ty }
    }
}

#[derive(Clone, Debug)]
pub(super) struct CastCheck<'db> {
    expr: ExprId,
    source_expr: ExprId,
    expr_ty: Ty<'db>,
    cast_ty: Ty<'db>,
}

impl<'db> CastCheck<'db> {
    pub(super) fn new(
        expr: ExprId,
        source_expr: ExprId,
        expr_ty: Ty<'db>,
        cast_ty: Ty<'db>,
    ) -> Self {
        Self { expr, source_expr, expr_ty, cast_ty }
    }

    pub(super) fn check(
        &mut self,
        ctx: &mut InferenceContext<'_, 'db>,
    ) -> Result<(), InferenceDiagnostic<'db>> {
        self.expr_ty = ctx.table.try_structurally_resolve_type(self.expr_ty);
        self.cast_ty = ctx.table.try_structurally_resolve_type(self.cast_ty);

        // This should always come first so that we apply the coercion, which impacts infer vars.
        if ctx
            .coerce(
                self.source_expr.into(),
                self.expr_ty,
                self.cast_ty,
                AllowTwoPhase::No,
                ExprIsRead::Yes,
            )
            .is_ok()
        {
            ctx.result.coercion_casts.insert(self.source_expr);
            return Ok(());
        }

        if self.expr_ty.references_non_lt_error() || self.cast_ty.references_non_lt_error() {
            return Ok(());
        }

        if !self.cast_ty.flags().contains(TypeFlags::HAS_TY_INFER)
            && !ctx.table.is_sized(self.cast_ty)
        {
            return Err(InferenceDiagnostic::CastToUnsized {
                expr: self.expr,
                cast_ty: self.cast_ty,
            });
        }

        // Chalk doesn't support trait upcasting and fails to solve some obvious goals
        // when the trait environment contains some recursive traits (See issue #18047)
        // We skip cast checks for such cases for now, until the next-gen solver.
        if contains_dyn_trait(self.cast_ty) {
            return Ok(());
        }

        self.do_check(ctx).map_err(|e| e.into_diagnostic(self.expr, self.expr_ty, self.cast_ty))
    }

    fn do_check(&self, ctx: &mut InferenceContext<'_, 'db>) -> Result<(), CastError> {
        let (t_from, t_cast) =
            match (CastTy::from_ty(ctx.db, self.expr_ty), CastTy::from_ty(ctx.db, self.cast_ty)) {
                (Some(t_from), Some(t_cast)) => (t_from, t_cast),
                (None, Some(t_cast)) => match self.expr_ty.kind() {
                    TyKind::FnDef(..) => {
                        let sig =
                            self.expr_ty.callable_sig(ctx.interner()).expect("FnDef had no sig");
                        let sig = ctx.table.normalize_associated_types_in(sig);
                        let fn_ptr = Ty::new_fn_ptr(ctx.interner(), sig);
                        if ctx
                            .coerce(
                                self.source_expr.into(),
                                self.expr_ty,
                                fn_ptr,
                                AllowTwoPhase::No,
                                ExprIsRead::Yes,
                            )
                            .is_ok()
                        {
                        } else {
                            return Err(CastError::IllegalCast);
                        }

                        (CastTy::FnPtr, t_cast)
                    }
                    TyKind::Ref(_, inner_ty, mutbl) => {
                        return match t_cast {
                            CastTy::Int(_) | CastTy::Float => match inner_ty.kind() {
                                TyKind::Int(_)
                                | TyKind::Uint(_)
                                | TyKind::Float(_)
                                | TyKind::Infer(InferTy::IntVar(_) | InferTy::FloatVar(_)) => {
                                    Err(CastError::NeedDeref)
                                }

                                _ => Err(CastError::NeedViaPtr),
                            },
                            // array-ptr-cast
                            CastTy::Ptr(t, m) => {
                                let t = ctx.table.try_structurally_resolve_type(t);
                                if !ctx.table.is_sized(t) {
                                    return Err(CastError::IllegalCast);
                                }
                                self.check_ref_cast(ctx, inner_ty, mutbl, t, m)
                            }
                            _ => Err(CastError::NonScalar),
                        };
                    }
                    _ => return Err(CastError::NonScalar),
                },
                _ => return Err(CastError::NonScalar),
            };

        // rustc checks whether the `expr_ty` is foreign adt with `non_exhaustive` sym

        match (t_from, t_cast) {
            (_, CastTy::Int(Int::CEnum) | CastTy::FnPtr) => Err(CastError::NonScalar),
            (_, CastTy::Int(Int::Bool)) => Err(CastError::CastToBool),
            (CastTy::Int(Int::U(UintTy::U8)), CastTy::Int(Int::Char)) => Ok(()),
            (_, CastTy::Int(Int::Char)) => Err(CastError::CastToChar),
            (CastTy::Int(Int::Bool | Int::CEnum | Int::Char), CastTy::Float) => {
                Err(CastError::NeedViaInt)
            }
            (CastTy::Int(Int::Bool | Int::CEnum | Int::Char) | CastTy::Float, CastTy::Ptr(..))
            | (CastTy::Ptr(..) | CastTy::FnPtr, CastTy::Float) => Err(CastError::IllegalCast),
            (CastTy::Ptr(src, _), CastTy::Ptr(dst, _)) => self.check_ptr_ptr_cast(ctx, src, dst),
            (CastTy::Ptr(src, _), CastTy::Int(_)) => self.check_ptr_addr_cast(ctx, src),
            (CastTy::Int(_), CastTy::Ptr(dst, _)) => self.check_addr_ptr_cast(ctx, dst),
            (CastTy::FnPtr, CastTy::Ptr(dst, _)) => self.check_fptr_ptr_cast(ctx, dst),
            (CastTy::Int(Int::CEnum), CastTy::Int(_)) => Ok(()),
            (CastTy::Int(Int::Char | Int::Bool), CastTy::Int(_)) => Ok(()),
            (CastTy::Int(_) | CastTy::Float, CastTy::Int(_) | CastTy::Float) => Ok(()),
            (CastTy::FnPtr, CastTy::Int(_)) => Ok(()),
        }
    }

    fn check_ref_cast(
        &self,
        ctx: &mut InferenceContext<'_, 'db>,
        t_expr: Ty<'db>,
        m_expr: Mutability,
        t_cast: Ty<'db>,
        m_cast: Mutability,
    ) -> Result<(), CastError> {
        // Mutability order is opposite to rustc. `Mut < Not`
        if m_expr <= m_cast
            && let TyKind::Array(ety, _) = t_expr.kind()
        {
            // Coerce to a raw pointer so that we generate RawPtr in MIR.
            let array_ptr_type = Ty::new_ptr(ctx.interner(), t_expr, m_expr);
            if ctx
                .coerce(
                    self.source_expr.into(),
                    self.expr_ty,
                    array_ptr_type,
                    AllowTwoPhase::No,
                    ExprIsRead::Yes,
                )
                .is_ok()
            {
            } else {
                never!(
                    "could not cast from reference to array to pointer to array ({:?} to {:?})",
                    self.expr_ty,
                    array_ptr_type
                );
            }

            // This is a less strict condition than rustc's `demand_eqtype`,
            // but false negative is better than false positive
            if ctx
                .coerce(self.source_expr.into(), ety, t_cast, AllowTwoPhase::No, ExprIsRead::Yes)
                .is_ok()
            {
                return Ok(());
            }
        }

        Err(CastError::IllegalCast)
    }

    fn check_ptr_ptr_cast(
        &self,
        ctx: &mut InferenceContext<'_, 'db>,
        src: Ty<'db>,
        dst: Ty<'db>,
    ) -> Result<(), CastError> {
        let src_kind = pointer_kind(src, ctx).map_err(|_| CastError::Unknown)?;
        let dst_kind = pointer_kind(dst, ctx).map_err(|_| CastError::Unknown)?;

        match (src_kind, dst_kind) {
            (Some(PointerKind::Error), _) | (_, Some(PointerKind::Error)) => Ok(()),
            // (_, None) => Err(CastError::UnknownCastPtrKind),
            // (None, _) => Err(CastError::UnknownExprPtrKind),
            (_, None) | (None, _) => Ok(()),
            (_, Some(PointerKind::Thin)) => Ok(()),
            (Some(PointerKind::Thin), _) => Err(CastError::SizedUnsizedCast),
            (Some(PointerKind::VTable(src_tty)), Some(PointerKind::VTable(dst_tty))) => {
                match (src_tty.principal_def_id(), dst_tty.principal_def_id()) {
                    (Some(src_principal), Some(dst_principal)) => {
                        if src_principal == dst_principal {
                            return Ok(());
                        }
                        let src_principal = ctx.db.trait_signature(src_principal.0);
                        let dst_principal = ctx.db.trait_signature(dst_principal.0);
                        if src_principal.flags.contains(TraitFlags::AUTO)
                            && dst_principal.flags.contains(TraitFlags::AUTO)
                        {
                            Ok(())
                        } else {
                            Err(CastError::DifferingKinds)
                        }
                    }
                    _ => Err(CastError::Unknown),
                }
            }
            (Some(src_kind), Some(dst_kind)) if src_kind == dst_kind => Ok(()),
            (_, _) => Err(CastError::DifferingKinds),
        }
    }

    fn check_ptr_addr_cast(
        &self,
        ctx: &mut InferenceContext<'_, 'db>,
        expr_ty: Ty<'db>,
    ) -> Result<(), CastError> {
        match pointer_kind(expr_ty, ctx).map_err(|_| CastError::Unknown)? {
            // None => Err(CastError::UnknownExprPtrKind),
            None => Ok(()),
            Some(PointerKind::Error) => Ok(()),
            Some(PointerKind::Thin) => Ok(()),
            _ => Err(CastError::NeedViaThinPtr),
        }
    }

    fn check_addr_ptr_cast(
        &self,
        ctx: &mut InferenceContext<'_, 'db>,
        cast_ty: Ty<'db>,
    ) -> Result<(), CastError> {
        match pointer_kind(cast_ty, ctx).map_err(|_| CastError::Unknown)? {
            // None => Err(CastError::UnknownCastPtrKind),
            None => Ok(()),
            Some(PointerKind::Error) => Ok(()),
            Some(PointerKind::Thin) => Ok(()),
            Some(PointerKind::VTable(_)) => Err(CastError::IntToFatCast),
            Some(PointerKind::Length) => Err(CastError::IntToFatCast),
            Some(PointerKind::OfAlias | PointerKind::OfParam(_)) => Err(CastError::IntToFatCast),
        }
    }

    fn check_fptr_ptr_cast(
        &self,
        ctx: &mut InferenceContext<'_, 'db>,
        cast_ty: Ty<'db>,
    ) -> Result<(), CastError> {
        match pointer_kind(cast_ty, ctx).map_err(|_| CastError::Unknown)? {
            // None => Err(CastError::UnknownCastPtrKind),
            None => Ok(()),
            Some(PointerKind::Error) => Ok(()),
            Some(PointerKind::Thin) => Ok(()),
            _ => Err(CastError::IllegalCast),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum PointerKind<'db> {
    // thin pointer
    Thin,
    // trait object
    VTable(BoundExistentialPredicates<'db>),
    // slice
    Length,
    OfAlias,
    OfParam(ParamTy),
    Error,
}

fn pointer_kind<'db>(
    ty: Ty<'db>,
    ctx: &mut InferenceContext<'_, 'db>,
) -> Result<Option<PointerKind<'db>>, ()> {
    let ty = ctx.table.try_structurally_resolve_type(ty);

    if ctx.table.is_sized(ty) {
        return Ok(Some(PointerKind::Thin));
    }

    match ty.kind() {
        TyKind::Slice(_) | TyKind::Str => Ok(Some(PointerKind::Length)),
        TyKind::Dynamic(bounds, _) => Ok(Some(PointerKind::VTable(bounds))),
        TyKind::Adt(adt_def, subst) => {
            let id = adt_def.def_id().0;
            let AdtId::StructId(id) = id else {
                never!("`{:?}` should be sized but is not?", ty);
                return Err(());
            };

            let struct_data = id.fields(ctx.db);
            if let Some((last_field, _)) = struct_data.fields().iter().last() {
                let last_field_ty =
                    ctx.db.field_types(id.into())[last_field].instantiate(ctx.interner(), subst);
                pointer_kind(last_field_ty, ctx)
            } else {
                Ok(Some(PointerKind::Thin))
            }
        }
        TyKind::Tuple(subst) => match subst.iter().next_back() {
            None => Ok(Some(PointerKind::Thin)),
            Some(ty) => pointer_kind(ty, ctx),
        },
        TyKind::Foreign(_) => Ok(Some(PointerKind::Thin)),
        TyKind::Alias(..) => Ok(Some(PointerKind::OfAlias)),
        TyKind::Error(_) => Ok(Some(PointerKind::Error)),
        TyKind::Param(idx) => Ok(Some(PointerKind::OfParam(idx))),
        TyKind::Bound(..) | TyKind::Placeholder(..) | TyKind::Infer(..) => Ok(None),
        TyKind::Int(_)
        | TyKind::Uint(_)
        | TyKind::Float(_)
        | TyKind::Bool
        | TyKind::Char
        | TyKind::Array(..)
        | TyKind::CoroutineWitness(..)
        | TyKind::RawPtr(..)
        | TyKind::Ref(..)
        | TyKind::FnDef(..)
        | TyKind::FnPtr(..)
        | TyKind::Closure(..)
        | TyKind::Coroutine(..)
        | TyKind::CoroutineClosure(..)
        | TyKind::Never => {
            never!("`{:?}` should be sized but is not?", ty);
            Err(())
        }
        TyKind::UnsafeBinder(..) | TyKind::Pat(..) => {
            never!("we don't produce these types: {ty:?}");
            Err(())
        }
    }
}

fn contains_dyn_trait<'db>(ty: Ty<'db>) -> bool {
    use std::ops::ControlFlow;

    use rustc_type_ir::{TypeSuperVisitable, TypeVisitable, TypeVisitor};

    struct DynTraitVisitor;

    impl<'db> TypeVisitor<DbInterner<'db>> for DynTraitVisitor {
        type Result = ControlFlow<()>;

        fn visit_ty(&mut self, ty: Ty<'db>) -> ControlFlow<()> {
            match ty.kind() {
                TyKind::Dynamic(..) => ControlFlow::Break(()),
                _ => ty.super_visit_with(self),
            }
        }
    }

    ty.visit_with(&mut DynTraitVisitor).is_break()
}
