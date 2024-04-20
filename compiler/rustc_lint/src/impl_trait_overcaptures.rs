use rustc_data_structures::fx::FxIndexSet;
use rustc_data_structures::unord::UnordSet;
use rustc_errors::{Applicability, LintDiagnostic};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_macros::LintDiagnostic;
use rustc_middle::middle::resolve_bound_vars::ResolvedArg;
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor,
};
use rustc_session::lint::FutureIncompatibilityReason;
use rustc_span::edition::Edition;
use rustc_span::{BytePos, Span};

use crate::fluent_generated as fluent;
use crate::{LateContext, LateLintPass};

// TODO: feature gate these too

declare_lint! {
    /// UwU
    pub IMPL_TRAIT_OVERCAPTURES,
    Allow,
    "will capture more lifetimes than possibly intended in edition 2024",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionSemanticsChange(Edition::Edition2024),
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2021/IntoIterator-for-arrays.html>",
    };
}

declare_lint! {
    /// UwU
    pub IMPL_TRAIT_REDUNDANT_CAPTURES,
    Warn,
    "uwu 2"
}

declare_lint_pass!(
    /// Lint for opaque types that will begin capturing in-scope but unmentioned lifetimes
    /// in edition 2024.
    ImplTraitOvercaptures => [IMPL_TRAIT_OVERCAPTURES, IMPL_TRAIT_REDUNDANT_CAPTURES]
);

impl<'tcx> LateLintPass<'tcx> for ImplTraitOvercaptures {
    fn check_item(&mut self, cx: &LateContext<'tcx>, it: &'tcx hir::Item<'tcx>) {
        match &it.kind {
            hir::ItemKind::Fn(..) => check_fn(cx.tcx, it.owner_id.def_id),
            _ => {}
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, it: &'tcx hir::ImplItem<'tcx>) {
        match &it.kind {
            hir::ImplItemKind::Fn(_, _) => check_fn(cx.tcx, it.owner_id.def_id),
            _ => {}
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, it: &'tcx hir::TraitItem<'tcx>) {
        match &it.kind {
            hir::TraitItemKind::Fn(_, _) => check_fn(cx.tcx, it.owner_id.def_id),
            _ => {}
        }
    }
}

fn check_fn(tcx: TyCtxt<'_>, parent_def_id: LocalDefId) {
    let sig = tcx.fn_sig(parent_def_id).instantiate_identity();

    let mut in_scope_parameters = FxIndexSet::default();
    // Populate the in_scope_parameters list first with all of the generics in scope
    let mut current_def_id = Some(parent_def_id.to_def_id());
    while let Some(def_id) = current_def_id {
        let generics = tcx.generics_of(def_id);
        for param in &generics.own_params {
            in_scope_parameters.insert(param.def_id);
        }
        current_def_id = generics.parent;
    }

    // Then visit the signature to walk through all the binders (incl. the late-bound
    // vars on the function itself, which we need to count too).
    sig.visit_with(&mut VisitOpaqueTypes {
        tcx,
        parent_def_id,
        in_scope_parameters,
        seen: Default::default(),
    });
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
                _ => {
                    self.tcx.dcx().span_delayed_bug(
                        self.tcx.def_span(self.parent_def_id),
                        format!("unsupported bound variable kind: {arg:?}"),
                    );
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
        if !t.has_aliases() {
            return;
        }

        if let ty::Alias(ty::Projection, opaque_ty) = *t.kind()
            && self.tcx.is_impl_trait_in_trait(opaque_ty.def_id)
        {
            // visit the opaque of the RPITIT
            self.tcx
                .type_of(opaque_ty.def_id)
                .instantiate(self.tcx, opaque_ty.args)
                .visit_with(self)
        } else if let ty::Alias(ty::Opaque, opaque_ty) = *t.kind()
            && let Some(opaque_def_id) = opaque_ty.def_id.as_local()
            // Don't recurse infinitely on an opaque
            && self.seen.insert(opaque_def_id)
            // If it's owned by this function
            && let opaque =
                self.tcx.hir_node_by_def_id(opaque_def_id).expect_item().expect_opaque_ty()
            && let hir::OpaqueTyOrigin::FnReturn(parent_def_id) = opaque.origin
            && parent_def_id == self.parent_def_id
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

            let opaque_span = self.tcx.def_span(opaque_def_id);
            let new_capture_rules =
                opaque_span.at_least_rust_2024() || self.tcx.features().lifetime_capture_rules_2024;

            // If we have uncaptured args, and if the opaque doesn't already have
            // `use<>` syntax on it, and we're < edition 2024, then warn the user.
            if !new_capture_rules
                && opaque.precise_capturing_args.is_none()
                && !uncaptured_spans.is_empty()
            {
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

                self.tcx.emit_node_span_lint(
                    IMPL_TRAIT_OVERCAPTURES,
                    self.tcx.local_def_id_to_hir_id(opaque_def_id),
                    opaque_span,
                    ImplTraitOvercapturesLint {
                        self_ty: t,
                        num_captured: uncaptured_spans.len(),
                        uncaptured_spans,
                        suggestion,
                    },
                );
            }
            // Otherwise, if we are edition 2024, have `use<>` syntax, and
            // have no uncaptured args, then we should warn to the user that
            // it's redundant to capture all args explicitly.
            else if new_capture_rules
                && let Some((captured_args, capturing_span)) = opaque.precise_capturing_args
            {
                let mut explicitly_captured = UnordSet::default();
                for arg in captured_args {
                    match self.tcx.named_bound_var(arg.hir_id()) {
                        Some(
                            ResolvedArg::EarlyBound(def_id) | ResolvedArg::LateBound(_, _, def_id),
                        ) => {
                            if self.tcx.def_kind(self.tcx.parent(def_id)) == DefKind::OpaqueTy {
                                let (ty::ReEarlyParam(ty::EarlyParamRegion { def_id, .. })
                                | ty::ReLateParam(ty::LateParamRegion {
                                    bound_region: ty::BoundRegionKind::BrNamed(def_id, _),
                                    ..
                                })) = self
                                    .tcx
                                    .map_opaque_lifetime_to_parent_lifetime(def_id.expect_local())
                                    .kind()
                                else {
                                    span_bug!(
                                        self.tcx.def_span(def_id),
                                        "variable should have been duplicated from a parent"
                                    );
                                };
                                explicitly_captured.insert(def_id);
                            } else {
                                explicitly_captured.insert(def_id);
                            }
                        }
                        _ => {
                            self.tcx.dcx().span_delayed_bug(
                                self.tcx().hir().span(arg.hir_id()),
                                "no valid for captured arg",
                            );
                        }
                    }
                }

                if self
                    .in_scope_parameters
                    .iter()
                    .all(|def_id| explicitly_captured.contains(def_id))
                {
                    self.tcx.emit_node_span_lint(
                        IMPL_TRAIT_REDUNDANT_CAPTURES,
                        self.tcx.local_def_id_to_hir_id(opaque_def_id),
                        opaque_span,
                        ImplTraitRedundantCapturesLint { capturing_span },
                    );
                }
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
    uncaptured_spans: Vec<Span>,
    self_ty: Ty<'tcx>,
    num_captured: usize,
    suggestion: Option<(String, Span)>,
}

impl<'a> LintDiagnostic<'a, ()> for ImplTraitOvercapturesLint<'_> {
    fn decorate_lint<'b>(self, diag: &'b mut rustc_errors::Diag<'a, ()>) {
        diag.arg("self_ty", self.self_ty.to_string())
            .arg("num_captured", self.num_captured)
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

#[derive(LintDiagnostic)]
#[diag(lint_impl_trait_redundant_captures)]
struct ImplTraitRedundantCapturesLint {
    #[suggestion(lint_suggestion, code = "", applicability = "machine-applicable")]
    capturing_span: Span,
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
