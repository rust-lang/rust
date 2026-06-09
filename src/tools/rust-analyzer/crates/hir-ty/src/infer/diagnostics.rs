//! This file contains the [`Diagnostics`] type used during inference,
//! and a wrapper around [`TyLoweringContext`] ([`InferenceTyLoweringContext`]) that replaces
//! it and takes care of diagnostics in inference.

use std::cell::{OnceCell, RefCell};
use std::ops::{Deref, DerefMut};

use either::Either;
use hir_def::expr_store::path::Path;
use hir_def::{ExpressionStoreOwnerId, GenericDefId};
use hir_def::{expr_store::ExpressionStore, type_ref::TypeRefId};
use hir_def::{hir::ExprOrPatId, resolver::Resolver};
use la_arena::{Idx, RawIdx};
use rustc_hash::FxHashMap;
use thin_vec::ThinVec;

use crate::{
    InferenceDiagnostic, InferenceTyDiagnosticSource, Span, TyLoweringDiagnostic,
    db::{AnonConstId, HirDatabase},
    generics::Generics,
    infer::unify::InferenceTable,
    lower::{
        ForbidParamsAfterReason, LifetimeElisionKind, TyLoweringContext, TyLoweringInferVarsCtx,
        path::{PathDiagnosticCallback, PathLoweringContext},
    },
    next_solver::{Const, Region, StoredTy, Ty},
};

// Unfortunately, this struct needs to use interior mutability (but we encapsulate it)
// because when lowering types and paths we hold a `TyLoweringContext` that holds a reference
// to our resolver and so we cannot have mutable reference, but we really want to have
// ability to dispatch diagnostics during this work otherwise the code becomes a complete mess.
#[derive(Debug, Default, Clone)]
pub(super) struct Diagnostics(RefCell<ThinVec<InferenceDiagnostic>>);

impl Diagnostics {
    pub(super) fn push(&self, diagnostic: InferenceDiagnostic) {
        self.0.borrow_mut().push(diagnostic);
    }

    fn push_ty_diagnostics(
        &self,
        source: InferenceTyDiagnosticSource,
        diagnostics: ThinVec<TyLoweringDiagnostic>,
    ) {
        self.0.borrow_mut().extend(
            diagnostics.into_iter().map(|diag| InferenceDiagnostic::TyDiagnostic { source, diag }),
        );
    }

    pub(super) fn finish(self) -> ThinVec<InferenceDiagnostic> {
        self.0.into_inner()
    }
}

pub(crate) struct PathDiagnosticCallbackData<'a> {
    node: ExprOrPatId,
    diagnostics: &'a Diagnostics,
}

pub(super) struct InferenceTyLoweringVarsCtx<'a, 'db> {
    pub(super) table: &'a mut InferenceTable<'db>,
    pub(super) type_of_type_placeholder: &'a mut FxHashMap<TypeRefId, StoredTy>,
}

impl<'db> TyLoweringInferVarsCtx<'db> for InferenceTyLoweringVarsCtx<'_, 'db> {
    fn next_ty_var(&mut self, span: Span) -> Ty<'db> {
        let ty = self.table.infer_ctxt.next_ty_var(span);

        if let Span::TypeRefId(type_ref) = span {
            self.type_of_type_placeholder.insert(type_ref, ty.store());
        }

        ty
    }
    fn next_const_var(&mut self, span: Span) -> Const<'db> {
        self.table.infer_ctxt.next_const_var(span)
    }
    fn next_region_var(&mut self, span: Span) -> Region<'db> {
        self.table.infer_ctxt.next_region_var(span)
    }

    fn as_table(&mut self) -> Option<&mut InferenceTable<'db>> {
        Some(self.table)
    }
}

pub(super) struct InferenceTyLoweringContext<'db, 'a> {
    ctx: TyLoweringContext<'db, 'a>,
    diagnostics: &'a Diagnostics,
    source: InferenceTyDiagnosticSource,
    defined_anon_consts: &'a RefCell<ThinVec<AnonConstId>>,
}

impl<'db, 'a> InferenceTyLoweringContext<'db, 'a> {
    #[inline]
    pub(super) fn new(
        db: &'db dyn HirDatabase,
        resolver: &'a Resolver<'db>,
        store: &'a ExpressionStore,
        diagnostics: &'a Diagnostics,
        source: InferenceTyDiagnosticSource,
        def: ExpressionStoreOwnerId,
        generic_def: GenericDefId,
        generics: &'a OnceCell<Generics<'db>>,
        lifetime_elision: LifetimeElisionKind<'db>,
        allow_using_generic_params: bool,
        infer_vars: Option<&'a mut dyn TyLoweringInferVarsCtx<'db>>,
        defined_anon_consts: &'a RefCell<ThinVec<AnonConstId>>,
    ) -> Self {
        let mut ctx = TyLoweringContext::new(
            db,
            resolver,
            store,
            def,
            generic_def,
            generics,
            lifetime_elision,
        )
        .with_infer_vars_behavior(infer_vars);
        if !allow_using_generic_params {
            ctx.forbid_params_after(0, ForbidParamsAfterReason::AnonConst);
        }
        Self { ctx, diagnostics, source, defined_anon_consts }
    }

    #[inline]
    pub(super) fn at_path<'b>(
        &'b mut self,
        path: &'b Path,
        node: ExprOrPatId,
    ) -> PathLoweringContext<'b, 'a, 'db> {
        let on_diagnostic = PathDiagnosticCallback {
            data: Either::Right(PathDiagnosticCallbackData { diagnostics: self.diagnostics, node }),
            callback: |data, _, diag| {
                let data = data.as_ref().right().unwrap();
                data.diagnostics
                    .push(InferenceDiagnostic::PathDiagnostic { node: data.node, diag });
            },
        };
        PathLoweringContext::new(&mut self.ctx, on_diagnostic, path)
    }

    #[inline]
    pub(super) fn at_path_forget_diagnostics<'b>(
        &'b mut self,
        path: &'b Path,
    ) -> PathLoweringContext<'b, 'a, 'db> {
        let on_diagnostic = PathDiagnosticCallback {
            data: Either::Right(PathDiagnosticCallbackData {
                diagnostics: self.diagnostics,
                node: ExprOrPatId::ExprId(Idx::from_raw(RawIdx::from_u32(0))),
            }),
            callback: |_data, _, _diag| {},
        };
        PathLoweringContext::new(&mut self.ctx, on_diagnostic, path)
    }

    #[inline]
    pub(super) fn forget_diagnostics(&mut self) {
        self.ctx.diagnostics.clear();
    }
}

impl<'db, 'a> Deref for InferenceTyLoweringContext<'db, 'a> {
    type Target = TyLoweringContext<'db, 'a>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}

impl DerefMut for InferenceTyLoweringContext<'_, '_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ctx
    }
}

impl Drop for InferenceTyLoweringContext<'_, '_> {
    #[inline]
    fn drop(&mut self) {
        self.diagnostics
            .push_ty_diagnostics(self.source, std::mem::take(&mut self.ctx.diagnostics));
        self.defined_anon_consts.borrow_mut().extend(self.ctx.defined_anon_consts.iter().copied());
    }
}
