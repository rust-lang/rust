//! This file contains the [`Diagnostics`] type used during inference,
//! and a wrapper around [`TyLoweringContext`] ([`InferenceTyLoweringContext`]) that replaces
//! it and takes care of diagnostics in inference.

use std::cell::RefCell;
use std::ops::{Deref, DerefMut};

use either::Either;
use hir_def::GenericDefId;
use hir_def::expr_store::ExpressionStore;
use hir_def::expr_store::path::Path;
use hir_def::{hir::ExprOrPatId, resolver::Resolver};
use la_arena::{Idx, RawIdx};

use crate::lower::LifetimeElisionKind;
use crate::{
    InferenceDiagnostic, InferenceTyDiagnosticSource, TyLoweringContext, TyLoweringDiagnostic,
    db::HirDatabase,
    lower::path::{PathDiagnosticCallback, PathLoweringContext},
};

// Unfortunately, this struct needs to use interior mutability (but we encapsulate it)
// because when lowering types and paths we hold a `TyLoweringContext` that holds a reference
// to our resolver and so we cannot have mutable reference, but we really want to have
// ability to dispatch diagnostics during this work otherwise the code becomes a complete mess.
#[derive(Debug, Default, Clone)]
pub(super) struct Diagnostics(RefCell<Vec<InferenceDiagnostic>>);

impl Diagnostics {
    pub(super) fn push(&self, diagnostic: InferenceDiagnostic) {
        self.0.borrow_mut().push(diagnostic);
    }

    fn push_ty_diagnostics(
        &self,
        source: InferenceTyDiagnosticSource,
        diagnostics: Vec<TyLoweringDiagnostic>,
    ) {
        self.0.borrow_mut().extend(
            diagnostics.into_iter().map(|diag| InferenceDiagnostic::TyDiagnostic { source, diag }),
        );
    }

    pub(super) fn finish(self) -> Vec<InferenceDiagnostic> {
        self.0.into_inner()
    }
}

pub(crate) struct PathDiagnosticCallbackData<'a> {
    node: ExprOrPatId,
    diagnostics: &'a Diagnostics,
}

pub(super) struct InferenceTyLoweringContext<'a> {
    ctx: TyLoweringContext<'a>,
    diagnostics: &'a Diagnostics,
    source: InferenceTyDiagnosticSource,
}

impl<'a> InferenceTyLoweringContext<'a> {
    #[inline]
    pub(super) fn new(
        db: &'a dyn HirDatabase,
        resolver: &'a Resolver<'_>,
        store: &'a ExpressionStore,
        diagnostics: &'a Diagnostics,
        source: InferenceTyDiagnosticSource,
        generic_def: GenericDefId,
        lifetime_elision: LifetimeElisionKind,
    ) -> Self {
        Self {
            ctx: TyLoweringContext::new(db, resolver, store, generic_def, lifetime_elision),
            diagnostics,
            source,
        }
    }

    #[inline]
    pub(super) fn at_path<'b>(
        &'b mut self,
        path: &'b Path,
        node: ExprOrPatId,
    ) -> PathLoweringContext<'b, 'a> {
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
    ) -> PathLoweringContext<'b, 'a> {
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

impl<'a> Deref for InferenceTyLoweringContext<'a> {
    type Target = TyLoweringContext<'a>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}

impl DerefMut for InferenceTyLoweringContext<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ctx
    }
}

impl Drop for InferenceTyLoweringContext<'_> {
    #[inline]
    fn drop(&mut self) {
        self.diagnostics
            .push_ty_diagnostics(self.source, std::mem::take(&mut self.ctx.diagnostics));
    }
}
