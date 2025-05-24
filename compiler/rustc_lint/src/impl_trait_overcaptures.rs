use std::assert_matches::debug_assert_matches;
use std::cell::LazyCell;

use rustc_data_structures::fx::{FxHashMap, FxIndexMap, FxIndexSet};
use rustc_data_structures::unord::UnordSet;
use rustc_errors::{LintDiagnostic, Subdiagnostic};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_macros::LintDiagnostic;
use rustc_middle::middle::resolve_bound_vars::ResolvedArg;
use rustc_middle::ty::relate::{
    Relate, RelateResult, TypeRelation, structurally_relate_consts, structurally_relate_tys,
};
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeFoldable, TypeSuperVisitable, TypeVisitable, TypeVisitableExt,
    TypeVisitor,
};
use rustc_middle::{bug, span_bug};
use rustc_session::lint::FutureIncompatibilityReason;
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::edition::Edition;
use rustc_span::{Span, Symbol};
use rustc_trait_selection::errors::{
    AddPreciseCapturingForOvercapture, impl_trait_overcapture_suggestion,
};
use rustc_trait_selection::regions::OutlivesEnvironmentBuildExt;
use rustc_trait_selection::traits::ObligationCtxt;

use crate::{LateContext, LateLintPass, fluent_generated as fluent};

declare_lint! {
    /// The `impl_trait_overcaptures` lint warns against cases where lifetime
    /// capture behavior will differ in edition 2024.
    ///
    /// In the 2024 edition, `impl Trait`s will capture all lifetimes in scope,
    /// rather than just the lifetimes that are mentioned in the bounds of the type.
    /// Often these sets are equal, but if not, it means that the `impl Trait` may
    /// cause erroneous borrow-checker errors.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail,edition2021
    /// # #![deny(impl_trait_overcaptures)]
    /// # use std::fmt::Display;
    /// let mut x = vec![];
    /// x.push(1);
    ///
    /// fn test(x: &Vec<i32>) -> impl Display {
    ///     x[0]
    /// }
    ///
    /// let element = test(&x);
    /// x.push(2);
    /// println!("{element}");
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// In edition < 2024, the returned `impl Display` doesn't capture the
    /// lifetime from the `&Vec<i32>`, so the vector can be mutably borrowed
    /// while the `impl Display` is live.
    ///
    /// To fix this, we can explicitly state that the `impl Display` doesn't
    /// capture any lifetimes, using `impl Display + use<>`.
    pub IMPL_TRAIT_OVERCAPTURES,
    Allow,
    "`impl Trait` will capture more lifetimes than possibly intended in edition 2024",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionSemanticsChange(Edition::Edition2024),
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2024/rpit-lifetime-capture.html>",
    };
}

declare_lint! {
    /// The `impl_trait_redundant_captures` lint warns against cases where use of the
    /// precise capturing `use<...>` syntax is not needed.
    ///
    /// In the 2024 edition, `impl Trait`s will capture all lifetimes in scope.
    /// If precise-capturing `use<...>` syntax is used, and the set of parameters
    /// that are captures are *equal* to the set of parameters in scope, then
    /// the syntax is redundant, and can be removed.
    ///
    /// ### Example
    ///
    /// ```rust,edition2024,compile_fail
    /// # #![deny(impl_trait_redundant_captures)]
    /// fn test<'a>(x: &'a i32) -> impl Sized + use<'a> { x }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// To fix this, remove the `use<'a>`, since the lifetime is already captured
    /// since it is in scope.
    pub IMPL_TRAIT_REDUNDANT_CAPTURES,
    Allow,
    "redundant precise-capturing `use<...>` syntax on an `impl Trait`",
}

declare_lint_pass!(
    /// Lint for opaque types that will begin capturing in-scope but unmentioned lifetimes
    /// in edition 2024.
    ImplTraitOvercaptures => [IMPL_TRAIT_OVERCAPTURES, IMPL_TRAIT_REDUNDANT_CAPTURES]
);

impl<'tcx> LateLintPass<'tcx> for ImplTraitOvercaptures {
    fn check_item(&mut self, cx: &LateContext<'tcx>, it: &'tcx hir::Item<'tcx>) {
        match &it.kind {
            hir::ItemKind::Fn { .. } => check_fn(cx.tcx, it.owner_id.def_id),
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

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
enum ParamKind {
    // Early-bound var.
    Early(Symbol, u32),
    // Late-bound var on function, not within a binder. We can capture these.
    Free(DefId, Symbol),
    // Late-bound var in a binder. We can't capture these yet.
    Late,
}

fn check_fn(tcx: TyCtxt<'_>, parent_def_id: LocalDefId) {
    let sig = tcx.fn_sig(parent_def_id).instantiate_identity();

    let mut in_scope_parameters = FxIndexMap::default();
    // Populate the in_scope_parameters list first with all of the generics in scope
    let mut current_def_id = Some(parent_def_id.to_def_id());
    while let Some(def_id) = current_def_id {
        let generics = tcx.generics_of(def_id);
        for param in &generics.own_params {
            in_scope_parameters.insert(param.def_id, ParamKind::Early(param.name, param.index));
        }
        current_def_id = generics.parent;
    }

    for bound_var in sig.bound_vars() {
        let ty::BoundVariableKind::Region(ty::BoundRegionKind::Named(def_id, name)) = bound_var
        else {
            span_bug!(tcx.def_span(parent_def_id), "unexpected non-lifetime binder on fn sig");
        };

        in_scope_parameters.insert(def_id, ParamKind::Free(def_id, name));
    }

    let sig = tcx.liberate_late_bound_regions(parent_def_id.to_def_id(), sig);

    // Then visit the signature to walk through all the binders (incl. the late-bound
    // vars on the function itself, which we need to count too).
    sig.visit_with(&mut VisitOpaqueTypes {
        tcx,
        parent_def_id,
        in_scope_parameters,
        seen: Default::default(),
        // Lazily compute these two, since they're likely a bit expensive.
        variances: LazyCell::new(|| {
            let mut functional_variances = FunctionalVariances {
                tcx,
                variances: FxHashMap::default(),
                ambient_variance: ty::Covariant,
                generics: tcx.generics_of(parent_def_id),
            };
            functional_variances.relate(sig, sig).unwrap();
            functional_variances.variances
        }),
        outlives_env: LazyCell::new(|| {
            let typing_env = ty::TypingEnv::non_body_analysis(tcx, parent_def_id);
            let (infcx, param_env) = tcx.infer_ctxt().build_with_typing_env(typing_env);
            let ocx = ObligationCtxt::new(&infcx);
            let assumed_wf_tys = ocx.assumed_wf_types(param_env, parent_def_id).unwrap_or_default();
            OutlivesEnvironment::new(&infcx, parent_def_id, param_env, assumed_wf_tys)
        }),
    });
}

struct VisitOpaqueTypes<'tcx, VarFn, OutlivesFn> {
    tcx: TyCtxt<'tcx>,
    parent_def_id: LocalDefId,
    in_scope_parameters: FxIndexMap<DefId, ParamKind>,
    variances: LazyCell<FxHashMap<DefId, ty::Variance>, VarFn>,
    outlives_env: LazyCell<OutlivesEnvironment<'tcx>, OutlivesFn>,
    seen: FxIndexSet<LocalDefId>,
}

impl<'tcx, VarFn, OutlivesFn> TypeVisitor<TyCtxt<'tcx>>
    for VisitOpaqueTypes<'tcx, VarFn, OutlivesFn>
where
    VarFn: FnOnce() -> FxHashMap<DefId, ty::Variance>,
    OutlivesFn: FnOnce() -> OutlivesEnvironment<'tcx>,
{
    fn visit_binder<T: TypeFoldable<TyCtxt<'tcx>>>(&mut self, t: &ty::Binder<'tcx, T>) {
        // When we get into a binder, we need to add its own bound vars to the scope.
        let mut added = vec![];
        for arg in t.bound_vars() {
            let arg: ty::BoundVariableKind = arg;
            match arg {
                ty::BoundVariableKind::Region(ty::BoundRegionKind::Named(def_id, ..))
                | ty::BoundVariableKind::Ty(ty::BoundTyKind::Param(def_id, _)) => {
                    added.push(def_id);
                    let unique = self.in_scope_parameters.insert(def_id, ParamKind::Late);
                    assert_eq!(unique, None);
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

    fn visit_ty(&mut self, t: Ty<'tcx>) {
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
                self.tcx.hir_node_by_def_id(opaque_def_id).expect_opaque_ty()
            // We want to recurse into RPITs and async fns, even though the latter
            // doesn't overcapture on its own, it may mention additional RPITs
            // in its bounds.
            && let hir::OpaqueTyOrigin::FnReturn { parent, .. }
                | hir::OpaqueTyOrigin::AsyncFn { parent, .. } = opaque.origin
            && parent == self.parent_def_id
        {
            let opaque_span = self.tcx.def_span(opaque_def_id);
            let new_capture_rules = opaque_span.at_least_rust_2024();
            if !new_capture_rules
                && !opaque.bounds.iter().any(|bound| matches!(bound, hir::GenericBound::Use(..)))
            {
                // Compute the set of args that are captured by the opaque...
                let mut captured = FxIndexSet::default();
                let mut captured_regions = FxIndexSet::default();
                let variances = self.tcx.variances_of(opaque_def_id);
                let mut current_def_id = Some(opaque_def_id.to_def_id());
                while let Some(def_id) = current_def_id {
                    let generics = self.tcx.generics_of(def_id);
                    for param in &generics.own_params {
                        // A param is captured if it's invariant.
                        if variances[param.index as usize] != ty::Invariant {
                            continue;
                        }

                        let arg = opaque_ty.args[param.index as usize];
                        // We need to turn all `ty::Param`/`ConstKind::Param` and
                        // `ReEarlyParam`/`ReBound` into def ids.
                        captured.insert(extract_def_id_from_arg(self.tcx, generics, arg));

                        captured_regions.extend(arg.as_region());
                    }
                    current_def_id = generics.parent;
                }

                // Compute the set of in scope params that are not captured.
                let mut uncaptured_args: FxIndexSet<_> = self
                    .in_scope_parameters
                    .iter()
                    .filter(|&(def_id, _)| !captured.contains(def_id))
                    .collect();
                // Remove the set of lifetimes that are in-scope that outlive some other captured
                // lifetime and are contravariant (i.e. covariant in argument position).
                uncaptured_args.retain(|&(def_id, kind)| {
                    let Some(ty::Bivariant | ty::Contravariant) = self.variances.get(def_id) else {
                        // Keep all covariant/invariant args. Also if variance is `None`,
                        // then that means it's either not a lifetime, or it didn't show up
                        // anywhere in the signature.
                        return true;
                    };
                    // We only computed variance of lifetimes...
                    debug_assert_matches!(self.tcx.def_kind(def_id), DefKind::LifetimeParam);
                    let uncaptured = match *kind {
                        ParamKind::Early(name, index) => ty::Region::new_early_param(
                            self.tcx,
                            ty::EarlyParamRegion { name, index },
                        ),
                        ParamKind::Free(def_id, name) => ty::Region::new_late_param(
                            self.tcx,
                            self.parent_def_id.to_def_id(),
                            ty::LateParamRegionKind::Named(def_id, name),
                        ),
                        // Totally ignore late bound args from binders.
                        ParamKind::Late => return true,
                    };
                    // Does this region outlive any captured region?
                    !captured_regions.iter().any(|r| {
                        self.outlives_env
                            .free_region_map()
                            .sub_free_regions(self.tcx, *r, uncaptured)
                    })
                });

                // If we have uncaptured args, and if the opaque doesn't already have
                // `use<>` syntax on it, and we're < edition 2024, then warn the user.
                if !uncaptured_args.is_empty() {
                    let suggestion = impl_trait_overcapture_suggestion(
                        self.tcx,
                        opaque_def_id,
                        self.parent_def_id,
                        captured,
                    );

                    let uncaptured_spans: Vec<_> = uncaptured_args
                        .into_iter()
                        .map(|(def_id, _)| self.tcx.def_span(def_id))
                        .collect();

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
            }

            // Otherwise, if we are edition 2024, have `use<>` syntax, and
            // have no uncaptured args, then we should warn to the user that
            // it's redundant to capture all args explicitly.
            if new_capture_rules
                && let Some((captured_args, capturing_span)) =
                    opaque.bounds.iter().find_map(|bound| match *bound {
                        hir::GenericBound::Use(a, s) => Some((a, s)),
                        _ => None,
                    })
            {
                let mut explicitly_captured = UnordSet::default();
                for arg in captured_args {
                    match self.tcx.named_bound_var(arg.hir_id()) {
                        Some(
                            ResolvedArg::EarlyBound(def_id) | ResolvedArg::LateBound(_, _, def_id),
                        ) => {
                            if self.tcx.def_kind(self.tcx.local_parent(def_id)) == DefKind::OpaqueTy
                            {
                                let def_id = self
                                    .tcx
                                    .map_opaque_lifetime_to_parent_lifetime(def_id)
                                    .opt_param_def_id(self.tcx, self.parent_def_id.to_def_id())
                                    .expect("variable should have been duplicated from parent");

                                explicitly_captured.insert(def_id);
                            } else {
                                explicitly_captured.insert(def_id.to_def_id());
                            }
                        }
                        _ => {
                            self.tcx.dcx().span_delayed_bug(
                                self.tcx.hir_span(arg.hir_id()),
                                "no valid for captured arg",
                            );
                        }
                    }
                }

                if self
                    .in_scope_parameters
                    .iter()
                    .all(|(def_id, _)| explicitly_captured.contains(def_id))
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
    suggestion: Option<AddPreciseCapturingForOvercapture>,
}

impl<'a> LintDiagnostic<'a, ()> for ImplTraitOvercapturesLint<'_> {
    fn decorate_lint<'b>(self, diag: &'b mut rustc_errors::Diag<'a, ()>) {
        diag.primary_message(fluent::lint_impl_trait_overcaptures);
        diag.arg("self_ty", self.self_ty.to_string())
            .arg("num_captured", self.num_captured)
            .span_note(self.uncaptured_spans, fluent::lint_note)
            .note(fluent::lint_note2);
        if let Some(suggestion) = self.suggestion {
            suggestion.add_to_diag(diag);
        }
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
    match arg.kind() {
        ty::GenericArgKind::Lifetime(re) => match re.kind() {
            ty::ReEarlyParam(ebr) => generics.region_param(ebr, tcx).def_id,
            ty::ReBound(
                _,
                ty::BoundRegion { kind: ty::BoundRegionKind::Named(def_id, ..), .. },
            )
            | ty::ReLateParam(ty::LateParamRegion {
                scope: _,
                kind: ty::LateParamRegionKind::Named(def_id, ..),
            }) => def_id,
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

/// Computes the variances of regions that appear in the type, but considering
/// late-bound regions too, which don't have their variance computed usually.
///
/// Like generalization, this is a unary operation implemented on top of the binary
/// relation infrastructure, mostly because it's much easier to have the relation
/// track the variance for you, rather than having to do it yourself.
struct FunctionalVariances<'tcx> {
    tcx: TyCtxt<'tcx>,
    variances: FxHashMap<DefId, ty::Variance>,
    ambient_variance: ty::Variance,
    generics: &'tcx ty::Generics,
}

impl<'tcx> TypeRelation<TyCtxt<'tcx>> for FunctionalVariances<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn relate_with_variance<T: Relate<TyCtxt<'tcx>>>(
        &mut self,
        variance: ty::Variance,
        _: ty::VarianceDiagInfo<TyCtxt<'tcx>>,
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T> {
        let old_variance = self.ambient_variance;
        self.ambient_variance = self.ambient_variance.xform(variance);
        self.relate(a, b).unwrap();
        self.ambient_variance = old_variance;
        Ok(a)
    }

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        structurally_relate_tys(self, a, b).unwrap();
        Ok(a)
    }

    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        _: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        let def_id = match a.kind() {
            ty::ReEarlyParam(ebr) => self.generics.region_param(ebr, self.tcx).def_id,
            ty::ReBound(
                _,
                ty::BoundRegion { kind: ty::BoundRegionKind::Named(def_id, ..), .. },
            )
            | ty::ReLateParam(ty::LateParamRegion {
                scope: _,
                kind: ty::LateParamRegionKind::Named(def_id, ..),
            }) => def_id,
            _ => {
                return Ok(a);
            }
        };

        if let Some(variance) = self.variances.get_mut(&def_id) {
            *variance = unify(*variance, self.ambient_variance);
        } else {
            self.variances.insert(def_id, self.ambient_variance);
        }

        Ok(a)
    }

    fn consts(
        &mut self,
        a: ty::Const<'tcx>,
        b: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        structurally_relate_consts(self, a, b).unwrap();
        Ok(a)
    }

    fn binders<T>(
        &mut self,
        a: ty::Binder<'tcx, T>,
        b: ty::Binder<'tcx, T>,
    ) -> RelateResult<'tcx, ty::Binder<'tcx, T>>
    where
        T: Relate<TyCtxt<'tcx>>,
    {
        self.relate(a.skip_binder(), b.skip_binder()).unwrap();
        Ok(a)
    }
}

/// What is the variance that satisfies the two variances?
fn unify(a: ty::Variance, b: ty::Variance) -> ty::Variance {
    match (a, b) {
        // Bivariance is lattice bottom.
        (ty::Bivariant, other) | (other, ty::Bivariant) => other,
        // Invariant is lattice top.
        (ty::Invariant, _) | (_, ty::Invariant) => ty::Invariant,
        // If type is required to be covariant and contravariant, then it's invariant.
        (ty::Contravariant, ty::Covariant) | (ty::Covariant, ty::Contravariant) => ty::Invariant,
        // Otherwise, co + co = co, contra + contra = contra.
        (ty::Contravariant, ty::Contravariant) => ty::Contravariant,
        (ty::Covariant, ty::Covariant) => ty::Covariant,
    }
}
