//! Type cast logic. Basically coercion + additional casts.

use hir_def::{AdtId, hir::ExprId, signatures::TraitFlags};
use rustc_ast_ir::Mutability;
use rustc_hash::FxHashSet;
use rustc_type_ir::{
    InferTy, TypeVisitableExt, UintTy, elaborate,
    error::TypeError,
    inherent::{AdtDef, BoundExistentialPredicates as _, IntoKind, Ty as _},
};
use stdx::never;

use crate::{
    InferenceDiagnostic,
    db::HirDatabase,
    infer::{AllowTwoPhase, InferenceContext, expr::ExprIsRead},
    next_solver::{
        BoundExistentialPredicates, ExistentialPredicate, ParamTy, Region, Ty, TyKind,
        infer::traits::ObligationCause,
    },
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
    IntToWideCast,
    NeedDeref,
    NeedViaPtr,
    NeedViaThinPtr,
    NeedViaInt,
    NonScalar,
    PtrPtrAddingAutoTraits,
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
    ) -> InferenceDiagnostic {
        InferenceDiagnostic::InvalidCast {
            expr,
            error: self,
            expr_ty: expr_ty.store(),
            cast_ty: cast_ty.store(),
        }
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
    ) -> Result<(), InferenceDiagnostic> {
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

        if !self.cast_ty.has_infer_types() && !ctx.table.is_sized(self.cast_ty) {
            return Err(InferenceDiagnostic::CastToUnsized {
                expr: self.expr,
                cast_ty: self.cast_ty.store(),
            });
        }

        self.do_check(ctx).map_err(|e| e.into_diagnostic(self.expr, self.expr_ty, self.cast_ty))
    }

    fn do_check(&self, ctx: &mut InferenceContext<'_, 'db>) -> Result<(), CastError> {
        let (t_from, t_cast) =
            match (CastTy::from_ty(ctx.db, self.expr_ty), CastTy::from_ty(ctx.db, self.cast_ty)) {
                (Some(t_from), Some(t_cast)) => (t_from, t_cast),
                (None, Some(t_cast)) => match self.expr_ty.kind() {
                    TyKind::FnDef(..) => {
                        // rustc calls `FnCtxt::normalize` on this but it's a no-op in next-solver
                        let sig = self.expr_ty.fn_sig(ctx.interner());
                        let fn_ptr = Ty::new_fn_ptr(ctx.interner(), sig);
                        match ctx.coerce(
                            self.source_expr.into(),
                            self.expr_ty,
                            fn_ptr,
                            AllowTwoPhase::No,
                            ExprIsRead::Yes,
                        ) {
                            Ok(_) => {}
                            Err(TypeError::IntrinsicCast) => {
                                return Err(CastError::IllegalCast);
                            }
                            Err(_) => {
                                return Err(CastError::NonScalar);
                            }
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
            // These types have invariants! can't cast into them.
            (_, CastTy::Int(Int::CEnum) | CastTy::FnPtr) => Err(CastError::NonScalar),

            // * -> Bool
            (_, CastTy::Int(Int::Bool)) => Err(CastError::CastToBool),

            // * -> Char
            (CastTy::Int(Int::U(UintTy::U8)), CastTy::Int(Int::Char)) => Ok(()), // u8-char-cast
            (_, CastTy::Int(Int::Char)) => Err(CastError::CastToChar),

            // prim -> float,ptr
            (CastTy::Int(Int::Bool | Int::CEnum | Int::Char), CastTy::Float) => {
                Err(CastError::NeedViaInt)
            }

            (CastTy::Int(Int::Bool | Int::CEnum | Int::Char) | CastTy::Float, CastTy::Ptr(..))
            | (CastTy::Ptr(..) | CastTy::FnPtr, CastTy::Float) => Err(CastError::IllegalCast),

            // ptr -> ptr
            (CastTy::Ptr(src, _), CastTy::Ptr(dst, _)) => self.check_ptr_ptr_cast(ctx, src, dst), // ptr-ptr-cast

            // // ptr-addr-cast
            (CastTy::Ptr(src, _), CastTy::Int(_)) => self.check_ptr_addr_cast(ctx, src),
            (CastTy::FnPtr, CastTy::Int(_)) => Ok(()),

            // addr-ptr-cast
            (CastTy::Int(_), CastTy::Ptr(dst, _)) => self.check_addr_ptr_cast(ctx, dst),

            // fn-ptr-cast
            (CastTy::FnPtr, CastTy::Ptr(dst, _)) => self.check_fptr_ptr_cast(ctx, dst),

            // prim -> prim
            (CastTy::Int(Int::CEnum), CastTy::Int(_)) => Ok(()),
            (CastTy::Int(Int::Char | Int::Bool), CastTy::Int(_)) => Ok(()),
            (CastTy::Int(_) | CastTy::Float, CastTy::Int(_) | CastTy::Float) => Ok(()),
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
        let t_expr = ctx.table.try_structurally_resolve_type(t_expr);
        let t_cast = ctx.table.try_structurally_resolve_type(t_cast);

        if m_expr >= m_cast
            && let TyKind::Array(ety, _) = t_expr.kind()
            && ctx.infcx().can_eq(ctx.table.param_env, ety, t_cast)
        {
            // Due to historical reasons we allow directly casting references of
            // arrays into raw pointers of their element type.

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

            // this will report a type mismatch if needed
            let _ = ctx.demand_eqtype(self.expr.into(), ety, t_cast);
            return Ok(());
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

            // Cast to thin pointer is OK
            (_, Some(PointerKind::Thin)) => Ok(()),

            // thin -> fat? report invalid cast (don't complain about vtable kinds)
            (Some(PointerKind::Thin), _) => Err(CastError::SizedUnsizedCast),

            // trait object -> trait object? need to do additional checks
            (Some(PointerKind::VTable(src_tty)), Some(PointerKind::VTable(dst_tty))) => {
                match (src_tty.principal_def_id(), dst_tty.principal_def_id()) {
                    // A<dyn Src<...> + SrcAuto> -> B<dyn Dst<...> + DstAuto>. need to make sure
                    // - `Src` and `Dst` traits are the same
                    // - traits have the same generic arguments
                    // - projections are the same
                    // - `SrcAuto` (+auto traits implied by `Src`) is a superset of `DstAuto`
                    //
                    // Note that trait upcasting goes through a different mechanism (`coerce_unsized`)
                    // and is unaffected by this check.
                    (Some(src_principal), Some(dst_principal)) => {
                        if src_principal == dst_principal {
                            return Ok(());
                        }

                        // We need to reconstruct trait object types.
                        // `m_src` and `m_dst` won't work for us here because they will potentially
                        // contain wrappers, which we do not care about.
                        //
                        // e.g. we want to allow `dyn T -> (dyn T,)`, etc.
                        //
                        // We also need to skip auto traits to emit an FCW and not an error.
                        let src_obj = Ty::new_dynamic(
                            ctx.interner(),
                            BoundExistentialPredicates::new_from_iter(
                                ctx.interner(),
                                src_tty.iter().filter(|pred| {
                                    !matches!(
                                        pred.skip_binder(),
                                        ExistentialPredicate::AutoTrait(_)
                                    )
                                }),
                            ),
                            Region::new_erased(ctx.interner()),
                        );
                        let dst_obj = Ty::new_dynamic(
                            ctx.interner(),
                            BoundExistentialPredicates::new_from_iter(
                                ctx.interner(),
                                dst_tty.iter().filter(|pred| {
                                    !matches!(
                                        pred.skip_binder(),
                                        ExistentialPredicate::AutoTrait(_)
                                    )
                                }),
                            ),
                            Region::new_erased(ctx.interner()),
                        );

                        // `dyn Src = dyn Dst`, this checks for matching traits/generics/projections
                        // This is `fcx.demand_eqtype`, but inlined to give a better error.
                        if ctx
                            .table
                            .at(&ObligationCause::dummy())
                            .eq(src_obj, dst_obj)
                            .map(|infer_ok| ctx.table.register_infer_ok(infer_ok))
                            .is_err()
                        {
                            return Err(CastError::DifferingKinds);
                        }

                        // Check that `SrcAuto` (+auto traits implied by `Src`) is a superset of `DstAuto`.
                        // Emit an FCW otherwise.
                        let src_auto: FxHashSet<_> = src_tty
                            .auto_traits()
                            .into_iter()
                            .chain(
                                elaborate::supertrait_def_ids(ctx.interner(), src_principal)
                                    .filter(|trait_| {
                                        ctx.db
                                            .trait_signature(trait_.0)
                                            .flags
                                            .contains(TraitFlags::AUTO)
                                    }),
                            )
                            .collect();

                        let added = dst_tty
                            .auto_traits()
                            .into_iter()
                            .any(|trait_| !src_auto.contains(&trait_));

                        if added {
                            return Err(CastError::PtrPtrAddingAutoTraits);
                        }

                        Ok(())
                    }

                    // dyn Auto -> dyn Auto'? ok.
                    (None, None) => Ok(()),

                    // dyn Trait -> dyn Auto? not ok (for now).
                    //
                    // Although dropping the principal is already allowed for unsizing coercions
                    // (e.g. `*const (dyn Trait + Auto)` to `*const dyn Auto`), dropping it is
                    // currently **NOT** allowed for (non-coercion) ptr-to-ptr casts (e.g
                    // `*const Foo` to `*const Bar` where `Foo` has a `dyn Trait + Auto` tail
                    // and `Bar` has a `dyn Auto` tail), because the underlying MIR operations
                    // currently work very differently:
                    //
                    // * A MIR unsizing coercion on raw pointers to trait objects (`*const dyn Src`
                    //   to `*const dyn Dst`) is currently equivalent to downcasting the source to
                    //   the concrete sized type that it was originally unsized from first (via a
                    //   ptr-to-ptr cast from `*const Src` to `*const T` with `T: Sized`) and then
                    //   unsizing this thin pointer to the target type (unsizing `*const T` to
                    //   `*const Dst`). In particular, this means that the pointer's metadata
                    //   (vtable) will semantically change, e.g. for const eval and miri, even
                    //   though the vtables will always be merged for codegen.
                    //
                    // * A MIR ptr-to-ptr cast is currently equivalent to a transmute and does not
                    //   change the pointer metadata (vtable) at all.
                    //
                    // In addition to this potentially surprising difference between coercion and
                    // non-coercion casts, casting away the principal with a MIR ptr-to-ptr cast
                    // is currently considered undefined behavior:
                    //
                    // As a validity invariant of pointers to trait objects, we currently require
                    // that the principal of the vtable in the pointer metadata exactly matches
                    // the principal of the pointee type, where "no principal" is also considered
                    // a kind of principal.
                    (Some(_), None) => Err(CastError::DifferingKinds),

                    // dyn Auto -> dyn Trait? not ok.
                    (None, Some(_)) => Err(CastError::DifferingKinds),
                }
            }

            // fat -> fat? metadata kinds must match
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
            Some(PointerKind::VTable(_)) => Err(CastError::IntToWideCast),
            Some(PointerKind::Length) => Err(CastError::IntToWideCast),
            Some(PointerKind::OfAlias | PointerKind::OfParam(_)) => Err(CastError::IntToWideCast),
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

/// The kind of pointer and associated metadata (thin, length or vtable) - we
/// only allow casts between wide pointers if their metadata have the same
/// kind.
#[derive(Debug, PartialEq, Eq)]
enum PointerKind<'db> {
    /// No metadata attached, ie pointer to sized type or foreign type
    Thin,
    /// A trait object
    VTable(BoundExistentialPredicates<'db>),
    /// Slice
    Length,
    /// The unsize info of this projection or opaque type
    OfAlias,
    /// The unsize info of this parameter
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
                let last_field_ty = ctx.db.field_types(id.into())[last_field]
                    .get()
                    .instantiate(ctx.interner(), subst);
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
