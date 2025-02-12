//! This file contains the [`Diagnostics`] type used during inference,
//! and a wrapper around [`TyLoweringContext`] ([`InferenceTyLoweringContext`]) that replaces
//! it and takes care of diagnostics in inference.

use std::cell::RefCell;
use std::ops::{Deref, DerefMut};

use hir_def::expr_store::HygieneId;
use hir_def::hir::ExprOrPatId;
use hir_def::path::{Path, PathSegment, PathSegments};
use hir_def::resolver::{ResolveValueResult, Resolver, TypeNs};
use hir_def::type_ref::TypesMap;
use hir_def::TypeOwnerId;

use crate::db::HirDatabase;
use crate::{
    InferenceDiagnostic, InferenceTyDiagnosticSource, Ty, TyLoweringContext, TyLoweringDiagnostic,
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

pub(super) struct InferenceTyLoweringContext<'a> {
    ctx: TyLoweringContext<'a>,
    diagnostics: &'a Diagnostics,
    source: InferenceTyDiagnosticSource,
}

impl<'a> InferenceTyLoweringContext<'a> {
    pub(super) fn new(
        db: &'a dyn HirDatabase,
        resolver: &'a Resolver,
        types_map: &'a TypesMap,
        owner: TypeOwnerId,
        diagnostics: &'a Diagnostics,
        source: InferenceTyDiagnosticSource,
    ) -> Self {
        Self { ctx: TyLoweringContext::new(db, resolver, types_map, owner), diagnostics, source }
    }

    pub(super) fn resolve_path_in_type_ns(
        &mut self,
        path: &Path,
        node: ExprOrPatId,
    ) -> Option<(TypeNs, Option<usize>)> {
        let diagnostics = self.diagnostics;
        self.ctx.resolve_path_in_type_ns(path, &mut |_, diag| {
            diagnostics.push(InferenceDiagnostic::PathDiagnostic { node, diag })
        })
    }

    pub(super) fn resolve_path_in_value_ns(
        &mut self,
        path: &Path,
        node: ExprOrPatId,
        hygiene_id: HygieneId,
    ) -> Option<ResolveValueResult> {
        let diagnostics = self.diagnostics;
        self.ctx.resolve_path_in_value_ns(path, hygiene_id, &mut |_, diag| {
            diagnostics.push(InferenceDiagnostic::PathDiagnostic { node, diag })
        })
    }

    pub(super) fn lower_partly_resolved_path(
        &mut self,
        node: ExprOrPatId,
        resolution: TypeNs,
        resolved_segment: PathSegment<'_>,
        remaining_segments: PathSegments<'_>,
        resolved_segment_idx: u32,
        infer_args: bool,
    ) -> (Ty, Option<TypeNs>) {
        let diagnostics = self.diagnostics;
        self.ctx.lower_partly_resolved_path(
            resolution,
            resolved_segment,
            remaining_segments,
            resolved_segment_idx,
            infer_args,
            &mut |_, diag| diagnostics.push(InferenceDiagnostic::PathDiagnostic { node, diag }),
        )
    }
}

impl<'a> Deref for InferenceTyLoweringContext<'a> {
    type Target = TyLoweringContext<'a>;

    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}

impl DerefMut for InferenceTyLoweringContext<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ctx
    }
}

impl Drop for InferenceTyLoweringContext<'_> {
    fn drop(&mut self) {
        self.diagnostics
            .push_ty_diagnostics(self.source, std::mem::take(&mut self.ctx.diagnostics));
    }
}
