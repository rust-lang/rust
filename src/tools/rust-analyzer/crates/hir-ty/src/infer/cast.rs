//! Type cast logic. Basically coercion + additional casts.

use chalk_ir::{Mutability, Scalar, TyVariableKind, UintTy};
use hir_def::{AdtId, hir::ExprId};
use stdx::never;

use crate::{
    Adjustment, Binders, DynTy, InferenceDiagnostic, Interner, PlaceholderIndex,
    QuantifiedWhereClauses, Ty, TyExt, TyKind, TypeFlags, WhereClause,
    infer::{coerce::CoerceNever, unify::InferenceTable},
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
pub(crate) enum CastTy {
    Int(Int),
    Float,
    FnPtr,
    Ptr(Ty, Mutability),
    // `DynStar` is Not supported yet in r-a
}

impl CastTy {
    pub(crate) fn from_ty(table: &mut InferenceTable<'_>, t: &Ty) -> Option<Self> {
        match t.kind(Interner) {
            TyKind::Scalar(Scalar::Bool) => Some(Self::Int(Int::Bool)),
            TyKind::Scalar(Scalar::Char) => Some(Self::Int(Int::Char)),
            TyKind::Scalar(Scalar::Int(_)) => Some(Self::Int(Int::I)),
            TyKind::Scalar(Scalar::Uint(it)) => Some(Self::Int(Int::U(*it))),
            TyKind::InferenceVar(_, TyVariableKind::Integer) => Some(Self::Int(Int::InferenceVar)),
            TyKind::InferenceVar(_, TyVariableKind::Float) => Some(Self::Float),
            TyKind::Scalar(Scalar::Float(_)) => Some(Self::Float),
            TyKind::Adt(..) => {
                let (AdtId::EnumId(id), _) = t.as_adt()? else {
                    return None;
                };
                let enum_data = id.enum_variants(table.db);
                if enum_data.is_payload_free(table.db) { Some(Self::Int(Int::CEnum)) } else { None }
            }
            TyKind::Raw(m, ty) => Some(Self::Ptr(ty.clone(), *m)),
            TyKind::Function(_) => Some(Self::FnPtr),
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
    fn into_diagnostic(self, expr: ExprId, expr_ty: Ty, cast_ty: Ty) -> InferenceDiagnostic {
        InferenceDiagnostic::InvalidCast { expr, error: self, expr_ty, cast_ty }
    }
}

#[derive(Clone, Debug)]
pub(super) struct CastCheck {
    expr: ExprId,
    source_expr: ExprId,
    expr_ty: Ty,
    cast_ty: Ty,
}

impl CastCheck {
    pub(super) fn new(expr: ExprId, source_expr: ExprId, expr_ty: Ty, cast_ty: Ty) -> Self {
        Self { expr, source_expr, expr_ty, cast_ty }
    }

    pub(super) fn check<F, G>(
        &mut self,
        table: &mut InferenceTable<'_>,
        apply_adjustments: &mut F,
        set_coercion_cast: &mut G,
    ) -> Result<(), InferenceDiagnostic>
    where
        F: FnMut(ExprId, Vec<Adjustment>),
        G: FnMut(ExprId),
    {
        self.expr_ty = table.eagerly_normalize_and_resolve_shallow_in(self.expr_ty.clone());
        self.cast_ty = table.eagerly_normalize_and_resolve_shallow_in(self.cast_ty.clone());

        if self.expr_ty.contains_unknown() || self.cast_ty.contains_unknown() {
            return Ok(());
        }

        if !self.cast_ty.data(Interner).flags.contains(TypeFlags::HAS_TY_INFER)
            && !table.is_sized(&self.cast_ty)
        {
            return Err(InferenceDiagnostic::CastToUnsized {
                expr: self.expr,
                cast_ty: self.cast_ty.clone(),
            });
        }

        // Chalk doesn't support trait upcasting and fails to solve some obvious goals
        // when the trait environment contains some recursive traits (See issue #18047)
        // We skip cast checks for such cases for now, until the next-gen solver.
        if contains_dyn_trait(&self.cast_ty) {
            return Ok(());
        }

        if let Ok((adj, _)) = table.coerce(&self.expr_ty, &self.cast_ty, CoerceNever::Yes) {
            apply_adjustments(self.source_expr, adj);
            set_coercion_cast(self.source_expr);
            return Ok(());
        }

        self.do_check(table, apply_adjustments)
            .map_err(|e| e.into_diagnostic(self.expr, self.expr_ty.clone(), self.cast_ty.clone()))
    }

    fn do_check<F>(
        &self,
        table: &mut InferenceTable<'_>,
        apply_adjustments: &mut F,
    ) -> Result<(), CastError>
    where
        F: FnMut(ExprId, Vec<Adjustment>),
    {
        let (t_from, t_cast) =
            match (CastTy::from_ty(table, &self.expr_ty), CastTy::from_ty(table, &self.cast_ty)) {
                (Some(t_from), Some(t_cast)) => (t_from, t_cast),
                (None, Some(t_cast)) => match self.expr_ty.kind(Interner) {
                    TyKind::FnDef(..) => {
                        let sig = self.expr_ty.callable_sig(table.db).expect("FnDef had no sig");
                        let sig = table.eagerly_normalize_and_resolve_shallow_in(sig);
                        let fn_ptr = TyKind::Function(sig.to_fn_ptr()).intern(Interner);
                        if let Ok((adj, _)) = table.coerce(&self.expr_ty, &fn_ptr, CoerceNever::Yes)
                        {
                            apply_adjustments(self.source_expr, adj);
                        } else {
                            return Err(CastError::IllegalCast);
                        }

                        (CastTy::FnPtr, t_cast)
                    }
                    TyKind::Ref(mutbl, _, inner_ty) => {
                        return match t_cast {
                            CastTy::Int(_) | CastTy::Float => match inner_ty.kind(Interner) {
                                TyKind::Scalar(
                                    Scalar::Int(_) | Scalar::Uint(_) | Scalar::Float(_),
                                )
                                | TyKind::InferenceVar(
                                    _,
                                    TyVariableKind::Integer | TyVariableKind::Float,
                                ) => Err(CastError::NeedDeref),

                                _ => Err(CastError::NeedViaPtr),
                            },
                            // array-ptr-cast
                            CastTy::Ptr(t, m) => {
                                let t = table.eagerly_normalize_and_resolve_shallow_in(t);
                                if !table.is_sized(&t) {
                                    return Err(CastError::IllegalCast);
                                }
                                self.check_ref_cast(
                                    table,
                                    inner_ty,
                                    *mutbl,
                                    &t,
                                    m,
                                    apply_adjustments,
                                )
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
            (CastTy::Ptr(src, _), CastTy::Ptr(dst, _)) => {
                self.check_ptr_ptr_cast(table, &src, &dst)
            }
            (CastTy::Ptr(src, _), CastTy::Int(_)) => self.check_ptr_addr_cast(table, &src),
            (CastTy::Int(_), CastTy::Ptr(dst, _)) => self.check_addr_ptr_cast(table, &dst),
            (CastTy::FnPtr, CastTy::Ptr(dst, _)) => self.check_fptr_ptr_cast(table, &dst),
            (CastTy::Int(Int::CEnum), CastTy::Int(_)) => Ok(()),
            (CastTy::Int(Int::Char | Int::Bool), CastTy::Int(_)) => Ok(()),
            (CastTy::Int(_) | CastTy::Float, CastTy::Int(_) | CastTy::Float) => Ok(()),
            (CastTy::FnPtr, CastTy::Int(_)) => Ok(()),
        }
    }

    fn check_ref_cast<F>(
        &self,
        table: &mut InferenceTable<'_>,
        t_expr: &Ty,
        m_expr: Mutability,
        t_cast: &Ty,
        m_cast: Mutability,
        apply_adjustments: &mut F,
    ) -> Result<(), CastError>
    where
        F: FnMut(ExprId, Vec<Adjustment>),
    {
        // Mutability order is opposite to rustc. `Mut < Not`
        if m_expr <= m_cast {
            if let TyKind::Array(ety, _) = t_expr.kind(Interner) {
                // Coerce to a raw pointer so that we generate RawPtr in MIR.
                let array_ptr_type = TyKind::Raw(m_expr, t_expr.clone()).intern(Interner);
                if let Ok((adj, _)) = table.coerce(&self.expr_ty, &array_ptr_type, CoerceNever::Yes)
                {
                    apply_adjustments(self.source_expr, adj);
                } else {
                    never!(
                        "could not cast from reference to array to pointer to array ({:?} to {:?})",
                        self.expr_ty,
                        array_ptr_type
                    );
                }

                // This is a less strict condition than rustc's `demand_eqtype`,
                // but false negative is better than false positive
                if table.coerce(ety, t_cast, CoerceNever::Yes).is_ok() {
                    return Ok(());
                }
            }
        }

        Err(CastError::IllegalCast)
    }

    fn check_ptr_ptr_cast(
        &self,
        table: &mut InferenceTable<'_>,
        src: &Ty,
        dst: &Ty,
    ) -> Result<(), CastError> {
        let src_kind = pointer_kind(src, table).map_err(|_| CastError::Unknown)?;
        let dst_kind = pointer_kind(dst, table).map_err(|_| CastError::Unknown)?;

        match (src_kind, dst_kind) {
            (Some(PointerKind::Error), _) | (_, Some(PointerKind::Error)) => Ok(()),
            // (_, None) => Err(CastError::UnknownCastPtrKind),
            // (None, _) => Err(CastError::UnknownExprPtrKind),
            (_, None) | (None, _) => Ok(()),
            (_, Some(PointerKind::Thin)) => Ok(()),
            (Some(PointerKind::Thin), _) => Err(CastError::SizedUnsizedCast),
            (Some(PointerKind::VTable(src_tty)), Some(PointerKind::VTable(dst_tty))) => {
                let principal = |tty: &Binders<QuantifiedWhereClauses>| {
                    tty.skip_binders().as_slice(Interner).first().and_then(|pred| {
                        if let WhereClause::Implemented(tr) = pred.skip_binders() {
                            Some(tr.trait_id)
                        } else {
                            None
                        }
                    })
                };
                match (principal(&src_tty), principal(&dst_tty)) {
                    (Some(src_principal), Some(dst_principal)) => {
                        if src_principal == dst_principal {
                            return Ok(());
                        }
                        let src_principal =
                            table.db.trait_datum(table.trait_env.krate, src_principal);
                        let dst_principal =
                            table.db.trait_datum(table.trait_env.krate, dst_principal);
                        if src_principal.is_auto_trait() && dst_principal.is_auto_trait() {
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
        table: &mut InferenceTable<'_>,
        expr_ty: &Ty,
    ) -> Result<(), CastError> {
        match pointer_kind(expr_ty, table).map_err(|_| CastError::Unknown)? {
            // None => Err(CastError::UnknownExprPtrKind),
            None => Ok(()),
            Some(PointerKind::Error) => Ok(()),
            Some(PointerKind::Thin) => Ok(()),
            _ => Err(CastError::NeedViaThinPtr),
        }
    }

    fn check_addr_ptr_cast(
        &self,
        table: &mut InferenceTable<'_>,
        cast_ty: &Ty,
    ) -> Result<(), CastError> {
        match pointer_kind(cast_ty, table).map_err(|_| CastError::Unknown)? {
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
        table: &mut InferenceTable<'_>,
        cast_ty: &Ty,
    ) -> Result<(), CastError> {
        match pointer_kind(cast_ty, table).map_err(|_| CastError::Unknown)? {
            // None => Err(CastError::UnknownCastPtrKind),
            None => Ok(()),
            Some(PointerKind::Error) => Ok(()),
            Some(PointerKind::Thin) => Ok(()),
            _ => Err(CastError::IllegalCast),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum PointerKind {
    // thin pointer
    Thin,
    // trait object
    VTable(Binders<QuantifiedWhereClauses>),
    // slice
    Length,
    OfAlias,
    OfParam(PlaceholderIndex),
    Error,
}

fn pointer_kind(ty: &Ty, table: &mut InferenceTable<'_>) -> Result<Option<PointerKind>, ()> {
    let ty = table.eagerly_normalize_and_resolve_shallow_in(ty.clone());

    if table.is_sized(&ty) {
        return Ok(Some(PointerKind::Thin));
    }

    match ty.kind(Interner) {
        TyKind::Slice(_) | TyKind::Str => Ok(Some(PointerKind::Length)),
        TyKind::Dyn(DynTy { bounds, .. }) => Ok(Some(PointerKind::VTable(bounds.clone()))),
        TyKind::Adt(chalk_ir::AdtId(id), subst) => {
            let AdtId::StructId(id) = *id else {
                never!("`{:?}` should be sized but is not?", ty);
                return Err(());
            };

            let struct_data = table.db.variant_fields(id.into());
            if let Some((last_field, _)) = struct_data.fields().iter().last() {
                let last_field_ty =
                    table.db.field_types(id.into())[last_field].clone().substitute(Interner, subst);
                pointer_kind(&last_field_ty, table)
            } else {
                Ok(Some(PointerKind::Thin))
            }
        }
        TyKind::Tuple(_, subst) => {
            match subst.iter(Interner).last().and_then(|arg| arg.ty(Interner)) {
                None => Ok(Some(PointerKind::Thin)),
                Some(ty) => pointer_kind(ty, table),
            }
        }
        TyKind::Foreign(_) => Ok(Some(PointerKind::Thin)),
        TyKind::Alias(_) | TyKind::AssociatedType(..) | TyKind::OpaqueType(..) => {
            Ok(Some(PointerKind::OfAlias))
        }
        TyKind::Error => Ok(Some(PointerKind::Error)),
        TyKind::Placeholder(idx) => Ok(Some(PointerKind::OfParam(*idx))),
        TyKind::BoundVar(_) | TyKind::InferenceVar(..) => Ok(None),
        TyKind::Scalar(_)
        | TyKind::Array(..)
        | TyKind::CoroutineWitness(..)
        | TyKind::Raw(..)
        | TyKind::Ref(..)
        | TyKind::FnDef(..)
        | TyKind::Function(_)
        | TyKind::Closure(..)
        | TyKind::Coroutine(..)
        | TyKind::Never => {
            never!("`{:?}` should be sized but is not?", ty);
            Err(())
        }
    }
}

fn contains_dyn_trait(ty: &Ty) -> bool {
    use std::ops::ControlFlow;

    use chalk_ir::{
        DebruijnIndex,
        visit::{TypeSuperVisitable, TypeVisitable, TypeVisitor},
    };

    struct DynTraitVisitor;

    impl TypeVisitor<Interner> for DynTraitVisitor {
        type BreakTy = ();

        fn as_dyn(&mut self) -> &mut dyn TypeVisitor<Interner, BreakTy = Self::BreakTy> {
            self
        }

        fn interner(&self) -> Interner {
            Interner
        }

        fn visit_ty(&mut self, ty: &Ty, outer_binder: DebruijnIndex) -> ControlFlow<Self::BreakTy> {
            match ty.kind(Interner) {
                TyKind::Dyn(_) => ControlFlow::Break(()),
                _ => ty.super_visit_with(self.as_dyn(), outer_binder),
            }
        }
    }

    ty.visit_with(DynTraitVisitor.as_dyn(), DebruijnIndex::INNERMOST).is_break()
}
