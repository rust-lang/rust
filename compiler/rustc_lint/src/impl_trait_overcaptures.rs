use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::{Applicability, LintDiagnostic};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit;
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor,
};
use rustc_session::lint::FutureIncompatibilityReason;
use rustc_span::edition::Edition;
use rustc_span::{BytePos, Span};

use crate::fluent_generated as fluent;
use crate::{LateContext, LateLintPass};

declare_lint! {
    /// UwU
    pub IMPL_TRAIT_OVERCAPTURES,
    Warn,
    "will capture more lifetimes than possibly intended in edition 2024",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionSemanticsChange(Edition::Edition2024),
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2021/IntoIterator-for-arrays.html>",
    };
}

declare_lint_pass!(
    /// Lint for opaque types that will begin capturing in-scope but unmentioned lifetimes
    /// in edition 2024.
    ImplTraitOvercaptures => [IMPL_TRAIT_OVERCAPTURES]
);

impl<'tcx> LateLintPass<'tcx> for ImplTraitOvercaptures {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        _: intravisit::FnKind<'tcx>,
        _: &'tcx hir::FnDecl<'tcx>,
        _: &'tcx hir::Body<'tcx>,
        _: Span,
        parent_def_id: LocalDefId,
    ) {
        match cx.tcx.def_kind(parent_def_id) {
            DefKind::AssocFn => {
                // RPITITs already capture all lifetimes in scope, so skip them.
                if matches!(
                    cx.tcx.def_kind(cx.tcx.local_parent(parent_def_id)),
                    DefKind::Trait | DefKind::Impl { of_trait: true }
                ) {
                    return;
                }
            }
            DefKind::Fn => {
                // All free functions need to check for overcaptures.
            }
            DefKind::Closure => return,
            kind => {
                unreachable!(
                    "expected function item, found {}",
                    kind.descr(parent_def_id.to_def_id())
                )
            }
        }

        let sig = cx.tcx.fn_sig(parent_def_id).instantiate_identity();

        let mut in_scope_parameters = FxIndexSet::default();
        // Populate the in_scope_parameters list first with all of the generics in scope
        let mut current_def_id = Some(parent_def_id.to_def_id());
        while let Some(def_id) = current_def_id {
            let generics = cx.tcx.generics_of(def_id);
            for param in &generics.params {
                in_scope_parameters.insert(param.def_id);
            }
            current_def_id = generics.parent;
        }

        // Then visit the signature to walk through all the binders (incl. the late-bound
        // vars on the function itself, which we need to count too).
        sig.visit_with(&mut VisitOpaqueTypes {
            tcx: cx.tcx,
            parent_def_id,
            in_scope_parameters,
            seen: Default::default(),
        });
    }
}

struct VisitOpaqueTypes<'tcx> {
    tcx: TyCtxt<'tcx>,
    parent_def_id: LocalDefId,
    in_scope_parameters: FxIndexSet<DefId>,
    seen: FxIndexSet<LocalDefId>,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for VisitOpaqueTypes<'tcx> {
    fn visit_binder<T: TypeVisitable<TyCtxt<'tcx>>>(
        &mut self,
        t: &ty::Binder<'tcx, T>,
    ) -> Self::Result {
        // When we get into a binder, we need to add its own bound vars to the scope.
        let mut added = vec![];
        for arg in t.bound_vars() {
            let arg: ty::BoundVariableKind = arg;
            match arg {
                ty::BoundVariableKind::Region(ty::BoundRegionKind::BrNamed(def_id, ..)) => {
                    added.push(def_id);
                    let unique = self.in_scope_parameters.insert(def_id);
                    assert!(unique);
                }
                ty::BoundVariableKind::Ty(_) => {
                    todo!("we don't support late-bound type params in `impl Trait`")
                }
                ty::BoundVariableKind::Region(..) => {
                    unreachable!("all AST-derived bound regions should have a name")
                }
                ty::BoundVariableKind::Const => {
                    unreachable!("non-lifetime binder consts are not allowed")
                }
            }
        }

        t.super_visit_with(self);

        // And remove them. The `shift_remove` should be `O(1)` since we're popping
        // them off from the end.
        for arg in added.into_iter().rev() {
            self.in_scope_parameters.shift_remove(&arg);
        }
    }

    fn visit_ty(&mut self, t: Ty<'tcx>) -> Self::Result {
        if !t.has_opaque_types() {
            return;
        }

        if let ty::Alias(ty::Opaque, opaque_ty) = *t.kind()
            && let Some(opaque_def_id) = opaque_ty.def_id.as_local()
            // Don't recurse infinitely on an opaque
            && self.seen.insert(opaque_def_id)
            // If it's owned by this function
            && let opaque =
                self.tcx.hir_node_by_def_id(opaque_def_id).expect_item().expect_opaque_ty()
            && let hir::OpaqueTyOrigin::FnReturn(parent_def_id) = opaque.origin
            && parent_def_id == self.parent_def_id
            // And if the opaque doesn't already have `use<>` syntax on it...
            && opaque.precise_capturing_args.is_none()
        {
            // Compute the set of args that are captured by the opaque...
            let mut captured = FxIndexSet::default();
            let variances = self.tcx.variances_of(opaque_def_id);
            let mut current_def_id = Some(opaque_def_id.to_def_id());
            while let Some(def_id) = current_def_id {
                let generics = self.tcx.generics_of(def_id);
                for param in &generics.own_params {
                    // A param is captured if it's invariant.
                    if variances[param.index as usize] != ty::Invariant {
                        continue;
                    }
                    // We need to turn all `ty::Param`/`ConstKind::Param` and
                    // `ReEarlyParam`/`ReBound` into def ids.
                    captured.insert(extract_def_id_from_arg(
                        self.tcx,
                        generics,
                        opaque_ty.args[param.index as usize],
                    ));
                }
                current_def_id = generics.parent;
            }

            // Compute the set of in scope params that are not captured. Get their spans,
            // since that's all we really care about them for emitting the diagnostic.
            let uncaptured_spans: Vec<_> = self
                .in_scope_parameters
                .iter()
                .filter(|def_id| !captured.contains(*def_id))
                .map(|def_id| self.tcx.def_span(def_id))
                .collect();

            if !uncaptured_spans.is_empty() {
                let opaque_span = self.tcx.def_span(opaque_def_id);

                let suggestion = if let Ok(snippet) =
                    self.tcx.sess.source_map().span_to_snippet(opaque_span)
                    && snippet.starts_with("impl ")
                {
                    let (lifetimes, others): (Vec<_>, Vec<_>) = captured
                        .into_iter()
                        .partition(|def_id| self.tcx.def_kind(*def_id) == DefKind::LifetimeParam);
                    // Take all lifetime params first, then all others (ty/ct).
                    let generics: Vec<_> = lifetimes
                        .into_iter()
                        .chain(others)
                        .map(|def_id| self.tcx.item_name(def_id).to_string())
                        .collect();
                    // Make sure that we're not trying to name any APITs
                    if generics.iter().all(|name| !name.starts_with("impl ")) {
                        Some((
                            format!(" use<{}>", generics.join(", ")),
                            opaque_span.with_lo(opaque_span.lo() + BytePos(4)).shrink_to_lo(),
                        ))
                    } else {
                        None
                    }
                } else {
                    None
                };

                self.tcx.emit_node_lint(
                    IMPL_TRAIT_OVERCAPTURES,
                    self.tcx.local_def_id_to_hir_id(opaque_def_id),
                    ImplTraitOvercapturesLint {
                        opaque_span,
                        self_ty: t,
                        num_captured: uncaptured_spans.len(),
                        uncaptured_spans,
                        suggestion,
                    },
                );
            }

            // Walk into the bounds of the opaque, too, since we want to get nested opaques
            // in this lint as well. Interestingly, one place that I expect this lint to fire
            // is for `impl for<'a> Bound<Out = impl Other>`, since `impl Other` will begin
            // to capture `'a` in e2024 (even though late-bound vars in opaques are not allowed).
            for clause in
                self.tcx.item_bounds(opaque_ty.def_id).iter_instantiated(self.tcx, opaque_ty.args)
            {
                clause.visit_with(self)
            }
        }

        t.super_visit_with(self);
    }
}

struct ImplTraitOvercapturesLint<'tcx> {
    opaque_span: Span,
    uncaptured_spans: Vec<Span>,
    self_ty: Ty<'tcx>,
    num_captured: usize,
    suggestion: Option<(String, Span)>,
}

impl<'a> LintDiagnostic<'a, ()> for ImplTraitOvercapturesLint<'_> {
    fn decorate_lint<'b>(self, diag: &'b mut rustc_errors::Diag<'a, ()>) {
        diag.arg("self_ty", self.self_ty.to_string())
            .arg("num_captured", self.num_captured)
            .span(self.opaque_span)
            .span_note(self.uncaptured_spans, fluent::lint_note)
            .note(fluent::lint_note2);
        if let Some((suggestion, span)) = self.suggestion {
            diag.span_suggestion(
                span,
                fluent::lint_suggestion,
                suggestion,
                Applicability::MachineApplicable,
            );
        }
    }

    fn msg(&self) -> rustc_errors::DiagMessage {
        fluent::lint_impl_trait_overcaptures
    }
}

fn extract_def_id_from_arg<'tcx>(
    tcx: TyCtxt<'tcx>,
    generics: &'tcx ty::Generics,
    arg: ty::GenericArg<'tcx>,
) -> DefId {
    match arg.unpack() {
        ty::GenericArgKind::Lifetime(re) => match *re {
            ty::ReEarlyParam(ebr) => generics.region_param(ebr, tcx).def_id,
            ty::ReBound(
                _,
                ty::BoundRegion { kind: ty::BoundRegionKind::BrNamed(def_id, ..), .. },
            ) => def_id,
            _ => unreachable!(),
        },
        ty::GenericArgKind::Type(ty) => {
            let ty::Param(param_ty) = *ty.kind() else {
                bug!();
            };
            generics.type_param(param_ty, tcx).def_id
        }
        ty::GenericArgKind::Const(ct) => {
            let ty::ConstKind::Param(param_ct) = ct.kind() else {
                bug!();
            };
            generics.const_param(param_ct, tcx).def_id
        }
    }
}
