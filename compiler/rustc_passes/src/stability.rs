//! A pass that annotates every item and method with its stability level,
//! propagating default levels lexically from parent to children ast nodes.

use std::num::NonZero;

use rustc_ast_lowering::stability::extern_abi_stability;
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::unord::{ExtendUnord, UnordMap, UnordSet};
use rustc_feature::{EnabledLangFeature, EnabledLibFeature};
use rustc_hir::attrs::{AttributeKind, DeprecatedSince};
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{CRATE_DEF_ID, LOCAL_CRATE, LocalDefId, LocalModDefId};
use rustc_hir::intravisit::{self, Visitor, VisitorExt};
use rustc_hir::{
    self as hir, AmbigArg, ConstStability, DefaultBodyStability, FieldDef, Item, ItemKind,
    Stability, StabilityLevel, StableSince, TraitRef, Ty, TyKind, UnstableReason,
    VERSION_PLACEHOLDER, Variant, find_attr,
};
use rustc_middle::hir::nested_filter;
use rustc_middle::middle::lib_features::{FeatureStability, LibFeatures};
use rustc_middle::middle::privacy::EffectiveVisibilities;
use rustc_middle::middle::stability::{AllowUnstable, Deprecated, DeprecationEntry, EvalResult};
use rustc_middle::query::{LocalCrate, Providers};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{AssocContainer, TyCtxt};
use rustc_session::lint;
use rustc_session::lint::builtin::{DEPRECATED, INEFFECTIVE_UNSTABLE_TRAIT_IMPL};
use rustc_span::{Span, Symbol, sym};
use tracing::instrument;

use crate::errors;

#[derive(PartialEq)]
enum AnnotationKind {
    /// Annotation is required if not inherited from unstable parents.
    Required,
    /// Annotation is useless, reject it.
    Prohibited,
    /// Deprecation annotation is useless, reject it. (Stability attribute is still required.)
    DeprecationProhibited,
    /// Annotation itself is useless, but it can be propagated to children.
    Container,
}

fn inherit_deprecation(def_kind: DefKind) -> bool {
    match def_kind {
        DefKind::LifetimeParam | DefKind::TyParam | DefKind::ConstParam => false,
        _ => true,
    }
}

fn inherit_const_stability(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    let def_kind = tcx.def_kind(def_id);
    match def_kind {
        DefKind::AssocFn | DefKind::AssocTy | DefKind::AssocConst => {
            match tcx.def_kind(tcx.local_parent(def_id)) {
                DefKind::Impl { of_trait: true } => true,
                _ => false,
            }
        }
        _ => false,
    }
}

fn annotation_kind(tcx: TyCtxt<'_>, def_id: LocalDefId) -> AnnotationKind {
    let def_kind = tcx.def_kind(def_id);
    match def_kind {
        // Inherent impls and foreign modules serve only as containers for other items,
        // they don't have their own stability. They still can be annotated as unstable
        // and propagate this unstability to children, but this annotation is completely
        // optional. They inherit stability from their parents when unannotated.
        DefKind::Impl { of_trait: false } | DefKind::ForeignMod => AnnotationKind::Container,
        DefKind::Impl { of_trait: true } => AnnotationKind::DeprecationProhibited,

        // Allow stability attributes on default generic arguments.
        DefKind::TyParam | DefKind::ConstParam => {
            match &tcx.hir_node_by_def_id(def_id).expect_generic_param().kind {
                hir::GenericParamKind::Type { default: Some(_), .. }
                | hir::GenericParamKind::Const { default: Some(_), .. } => {
                    AnnotationKind::Container
                }
                _ => AnnotationKind::Prohibited,
            }
        }

        // Impl items in trait impls cannot have stability.
        DefKind::AssocTy | DefKind::AssocFn | DefKind::AssocConst => {
            match tcx.def_kind(tcx.local_parent(def_id)) {
                DefKind::Impl { of_trait: true } => AnnotationKind::Prohibited,
                _ => AnnotationKind::Required,
            }
        }

        _ => AnnotationKind::Required,
    }
}

fn lookup_deprecation_entry(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Option<DeprecationEntry> {
    let attrs = tcx.hir_attrs(tcx.local_def_id_to_hir_id(def_id));
    let depr = find_attr!(attrs,
        AttributeKind::Deprecation { deprecation, span: _ } => *deprecation
    );

    let Some(depr) = depr else {
        if inherit_deprecation(tcx.def_kind(def_id)) {
            let parent_id = tcx.opt_local_parent(def_id)?;
            let parent_depr = tcx.lookup_deprecation_entry(parent_id)?;
            return Some(parent_depr);
        }

        return None;
    };

    // `Deprecation` is just two pointers, no need to intern it
    Some(DeprecationEntry::local(depr, def_id))
}

fn inherit_stability(def_kind: DefKind) -> bool {
    match def_kind {
        DefKind::Field | DefKind::Variant | DefKind::Ctor(..) => true,
        _ => false,
    }
}

/// If the `-Z force-unstable-if-unmarked` flag is passed then we provide
/// a parent stability annotation which indicates that this is private
/// with the `rustc_private` feature. This is intended for use when
/// compiling library and `rustc_*` crates themselves so we can leverage crates.io
/// while maintaining the invariant that all sysroot crates are unstable
/// by default and are unable to be used.
const FORCE_UNSTABLE: Stability = Stability {
    level: StabilityLevel::Unstable {
        reason: UnstableReason::Default,
        issue: NonZero::new(27812),
        is_soft: false,
        implied_by: None,
        old_name: None,
    },
    feature: sym::rustc_private,
};

#[instrument(level = "debug", skip(tcx))]
fn lookup_stability(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Option<Stability> {
    // Propagate unstability. This can happen even for non-staged-api crates in case
    // -Zforce-unstable-if-unmarked is set.
    if !tcx.features().staged_api() {
        if !tcx.sess.opts.unstable_opts.force_unstable_if_unmarked {
            return None;
        }

        let Some(parent) = tcx.opt_local_parent(def_id) else { return Some(FORCE_UNSTABLE) };

        if inherit_deprecation(tcx.def_kind(def_id)) {
            let parent = tcx.lookup_stability(parent)?;
            if parent.is_unstable() {
                return Some(parent);
            }
        }

        return None;
    }

    // # Regular stability
    let attrs = tcx.hir_attrs(tcx.local_def_id_to_hir_id(def_id));
    let stab = find_attr!(attrs, AttributeKind::Stability { stability, span: _ } => *stability);

    if let Some(stab) = stab {
        return Some(stab);
    }

    if inherit_deprecation(tcx.def_kind(def_id)) {
        let Some(parent) = tcx.opt_local_parent(def_id) else {
            return tcx
                .sess
                .opts
                .unstable_opts
                .force_unstable_if_unmarked
                .then_some(FORCE_UNSTABLE);
        };
        let parent = tcx.lookup_stability(parent)?;
        if parent.is_unstable() || inherit_stability(tcx.def_kind(def_id)) {
            return Some(parent);
        }
    }

    None
}

#[instrument(level = "debug", skip(tcx))]
fn lookup_default_body_stability(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
) -> Option<DefaultBodyStability> {
    if !tcx.features().staged_api() {
        return None;
    }

    let attrs = tcx.hir_attrs(tcx.local_def_id_to_hir_id(def_id));
    // FIXME: check that this item can have body stability
    find_attr!(attrs, AttributeKind::BodyStability { stability, .. } => *stability)
}

#[instrument(level = "debug", skip(tcx))]
fn lookup_const_stability(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Option<ConstStability> {
    if !tcx.features().staged_api() {
        // Propagate unstability. This can happen even for non-staged-api crates in case
        // -Zforce-unstable-if-unmarked is set.
        if inherit_deprecation(tcx.def_kind(def_id)) {
            let parent = tcx.opt_local_parent(def_id)?;
            let parent_stab = tcx.lookup_stability(parent)?;
            if parent_stab.is_unstable()
                && let Some(fn_sig) = tcx.hir_node_by_def_id(def_id).fn_sig()
                && fn_sig.header.is_const()
            {
                let attrs = tcx.hir_attrs(tcx.local_def_id_to_hir_id(def_id));
                let const_stability_indirect =
                    find_attr!(attrs, AttributeKind::ConstStabilityIndirect);
                return Some(ConstStability::unmarked(const_stability_indirect, parent_stab));
            }
        }

        return None;
    }

    let attrs = tcx.hir_attrs(tcx.local_def_id_to_hir_id(def_id));
    let const_stability_indirect = find_attr!(attrs, AttributeKind::ConstStabilityIndirect);
    let const_stab =
        find_attr!(attrs, AttributeKind::ConstStability { stability, span: _ } => *stability);

    // After checking the immediate attributes, get rid of the span and compute implied
    // const stability: inherit feature gate from regular stability.
    let mut const_stab = const_stab
        .map(|const_stab| ConstStability::from_partial(const_stab, const_stability_indirect));

    // If this is a const fn but not annotated with stability markers, see if we can inherit
    // regular stability.
    if let Some(fn_sig) = tcx.hir_node_by_def_id(def_id).fn_sig()
        && fn_sig.header.is_const()
        && const_stab.is_none()
        // We only ever inherit unstable features.
        && let Some(inherit_regular_stab) = tcx.lookup_stability(def_id)
        && inherit_regular_stab.is_unstable()
    {
        const_stab = Some(ConstStability {
            // We subject these implicitly-const functions to recursive const stability.
            const_stable_indirect: true,
            promotable: false,
            level: inherit_regular_stab.level,
            feature: inherit_regular_stab.feature,
        });
    }

    if let Some(const_stab) = const_stab {
        return Some(const_stab);
    }

    // `impl const Trait for Type` items forward their const stability to their immediate children.
    // FIXME(const_trait_impl): how is this supposed to interact with `#[rustc_const_stable_indirect]`?
    // Currently, once that is set, we do not inherit anything from the parent any more.
    if inherit_const_stability(tcx, def_id) {
        let parent = tcx.opt_local_parent(def_id)?;
        let parent = tcx.lookup_const_stability(parent)?;
        if parent.is_const_unstable() {
            return Some(parent);
        }
    }

    None
}

fn stability_implications(tcx: TyCtxt<'_>, LocalCrate: LocalCrate) -> UnordMap<Symbol, Symbol> {
    let mut implications = UnordMap::default();

    let mut register_implication = |def_id| {
        if let Some(stability) = tcx.lookup_stability(def_id)
            && let StabilityLevel::Unstable { implied_by: Some(implied_by), .. } = stability.level
        {
            implications.insert(implied_by, stability.feature);
        }

        if let Some(stability) = tcx.lookup_const_stability(def_id)
            && let StabilityLevel::Unstable { implied_by: Some(implied_by), .. } = stability.level
        {
            implications.insert(implied_by, stability.feature);
        }
    };

    if tcx.features().staged_api() {
        register_implication(CRATE_DEF_ID);
        for def_id in tcx.hir_crate_items(()).definitions() {
            register_implication(def_id);
            let def_kind = tcx.def_kind(def_id);
            if def_kind.is_adt() {
                let adt = tcx.adt_def(def_id);
                for variant in adt.variants() {
                    if variant.def_id != def_id.to_def_id() {
                        register_implication(variant.def_id.expect_local());
                    }
                    for field in &variant.fields {
                        register_implication(field.did.expect_local());
                    }
                    if let Some(ctor_def_id) = variant.ctor_def_id() {
                        register_implication(ctor_def_id.expect_local())
                    }
                }
            }
            if def_kind.has_generics() {
                for param in tcx.generics_of(def_id).own_params.iter() {
                    register_implication(param.def_id.expect_local())
                }
            }
        }
    }

    implications
}

struct MissingStabilityAnnotations<'tcx> {
    tcx: TyCtxt<'tcx>,
    effective_visibilities: &'tcx EffectiveVisibilities,
}

impl<'tcx> MissingStabilityAnnotations<'tcx> {
    /// Verify that deprecation and stability attributes make sense with one another.
    #[instrument(level = "trace", skip(self))]
    fn check_compatible_stability(&self, def_id: LocalDefId) {
        if !self.tcx.features().staged_api() {
            return;
        }

        let depr = self.tcx.lookup_deprecation_entry(def_id);
        let stab = self.tcx.lookup_stability(def_id);
        let const_stab = self.tcx.lookup_const_stability(def_id);

        macro_rules! find_attr_span {
            ($name:ident) => {{
                let attrs = self.tcx.hir_attrs(self.tcx.local_def_id_to_hir_id(def_id));
                find_attr!(attrs, AttributeKind::$name { span, .. } => *span)
            }}
        }

        if stab.is_none()
            && depr.map_or(false, |d| d.attr.is_since_rustc_version())
            && let Some(span) = find_attr_span!(Deprecation)
        {
            self.tcx.dcx().emit_err(errors::DeprecatedAttribute { span });
        }

        if let Some(stab) = stab {
            // Error if prohibited, or can't inherit anything from a container.
            let kind = annotation_kind(self.tcx, def_id);
            if kind == AnnotationKind::Prohibited
                || (kind == AnnotationKind::Container && stab.level.is_stable() && depr.is_some())
            {
                if let Some(span) = find_attr_span!(Stability) {
                    let item_sp = self.tcx.def_span(def_id);
                    self.tcx.dcx().emit_err(errors::UselessStability { span, item_sp });
                }
            }

            // Check if deprecated_since < stable_since. If it is,
            // this is *almost surely* an accident.
            if let Some(depr) = depr
                && let DeprecatedSince::RustcVersion(dep_since) = depr.attr.since
                && let StabilityLevel::Stable { since: stab_since, .. } = stab.level
                && let Some(span) = find_attr_span!(Stability)
            {
                let item_sp = self.tcx.def_span(def_id);
                match stab_since {
                    StableSince::Current => {
                        self.tcx
                            .dcx()
                            .emit_err(errors::CannotStabilizeDeprecated { span, item_sp });
                    }
                    StableSince::Version(stab_since) => {
                        if dep_since < stab_since {
                            self.tcx
                                .dcx()
                                .emit_err(errors::CannotStabilizeDeprecated { span, item_sp });
                        }
                    }
                    StableSince::Err(_) => {
                        // An error already reported. Assume the unparseable stabilization
                        // version is older than the deprecation version.
                    }
                }
            }
        }

        // If the current node is a function with const stability attributes (directly given or
        // implied), check if the function/method is const or the parent impl block is const.
        let fn_sig = self.tcx.hir_node_by_def_id(def_id).fn_sig();
        if let Some(fn_sig) = fn_sig
            && !fn_sig.header.is_const()
            && const_stab.is_some()
            && find_attr_span!(ConstStability).is_some()
        {
            self.tcx.dcx().emit_err(errors::MissingConstErr { fn_sig_span: fn_sig.span });
        }

        // If this is marked const *stable*, it must also be regular-stable.
        if let Some(const_stab) = const_stab
            && let Some(fn_sig) = fn_sig
            && const_stab.is_const_stable()
            && !stab.is_some_and(|s| s.is_stable())
            && let Some(const_span) = find_attr_span!(ConstStability)
        {
            self.tcx
                .dcx()
                .emit_err(errors::ConstStableNotStable { fn_sig_span: fn_sig.span, const_span });
        }

        if let Some(stab) = &const_stab
            && stab.is_const_stable()
            && stab.const_stable_indirect
            && let Some(span) = find_attr_span!(ConstStability)
        {
            self.tcx.dcx().emit_err(errors::RustcConstStableIndirectPairing { span });
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn check_missing_stability(&self, def_id: LocalDefId) {
        let stab = self.tcx.lookup_stability(def_id);
        self.tcx.ensure_ok().lookup_const_stability(def_id);
        if !self.tcx.sess.is_test_crate()
            && stab.is_none()
            && self.effective_visibilities.is_reachable(def_id)
        {
            let descr = self.tcx.def_descr(def_id.to_def_id());
            let span = self.tcx.def_span(def_id);
            self.tcx.dcx().emit_err(errors::MissingStabilityAttr { span, descr });
        }
    }

    fn check_missing_const_stability(&self, def_id: LocalDefId) {
        let is_const = self.tcx.is_const_fn(def_id.to_def_id())
            || (self.tcx.def_kind(def_id.to_def_id()) == DefKind::Trait
                && self.tcx.is_const_trait(def_id.to_def_id()));

        // Reachable const fn/trait must have a stability attribute.
        if is_const
            && self.effective_visibilities.is_reachable(def_id)
            && self.tcx.lookup_const_stability(def_id).is_none()
        {
            let span = self.tcx.def_span(def_id);
            let descr = self.tcx.def_descr(def_id.to_def_id());
            self.tcx.dcx().emit_err(errors::MissingConstStabAttr { span, descr });
        }
    }
}

impl<'tcx> Visitor<'tcx> for MissingStabilityAnnotations<'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }

    fn visit_item(&mut self, i: &'tcx Item<'tcx>) {
        self.check_compatible_stability(i.owner_id.def_id);

        // Inherent impls and foreign modules serve only as containers for other items,
        // they don't have their own stability. They still can be annotated as unstable
        // and propagate this instability to children, but this annotation is completely
        // optional. They inherit stability from their parents when unannotated.
        if !matches!(
            i.kind,
            hir::ItemKind::Impl(hir::Impl { of_trait: None, .. })
                | hir::ItemKind::ForeignMod { .. }
        ) {
            self.check_missing_stability(i.owner_id.def_id);
        }

        // Ensure stable `const fn` have a const stability attribute.
        self.check_missing_const_stability(i.owner_id.def_id);

        intravisit::walk_item(self, i)
    }

    fn visit_trait_item(&mut self, ti: &'tcx hir::TraitItem<'tcx>) {
        self.check_compatible_stability(ti.owner_id.def_id);
        self.check_missing_stability(ti.owner_id.def_id);
        intravisit::walk_trait_item(self, ti);
    }

    fn visit_impl_item(&mut self, ii: &'tcx hir::ImplItem<'tcx>) {
        self.check_compatible_stability(ii.owner_id.def_id);
        if let hir::ImplItemImplKind::Inherent { .. } = ii.impl_kind {
            self.check_missing_stability(ii.owner_id.def_id);
            self.check_missing_const_stability(ii.owner_id.def_id);
        }
        intravisit::walk_impl_item(self, ii);
    }

    fn visit_variant(&mut self, var: &'tcx Variant<'tcx>) {
        self.check_compatible_stability(var.def_id);
        self.check_missing_stability(var.def_id);
        if let Some(ctor_def_id) = var.data.ctor_def_id() {
            self.check_missing_stability(ctor_def_id);
        }
        intravisit::walk_variant(self, var);
    }

    fn visit_field_def(&mut self, s: &'tcx FieldDef<'tcx>) {
        self.check_compatible_stability(s.def_id);
        self.check_missing_stability(s.def_id);
        intravisit::walk_field_def(self, s);
    }

    fn visit_foreign_item(&mut self, i: &'tcx hir::ForeignItem<'tcx>) {
        self.check_compatible_stability(i.owner_id.def_id);
        self.check_missing_stability(i.owner_id.def_id);
        intravisit::walk_foreign_item(self, i);
    }

    fn visit_generic_param(&mut self, p: &'tcx hir::GenericParam<'tcx>) {
        self.check_compatible_stability(p.def_id);
        // Note that we don't need to `check_missing_stability` for default generic parameters,
        // as we assume that any default generic parameters without attributes are automatically
        // stable (assuming they have not inherited instability from their parent).
        intravisit::walk_generic_param(self, p);
    }
}

/// Cross-references the feature names of unstable APIs with enabled
/// features and possibly prints errors.
fn check_mod_unstable_api_usage(tcx: TyCtxt<'_>, module_def_id: LocalModDefId) {
    tcx.hir_visit_item_likes_in_module(module_def_id, &mut Checker { tcx });

    let is_staged_api =
        tcx.sess.opts.unstable_opts.force_unstable_if_unmarked || tcx.features().staged_api();
    if is_staged_api {
        let effective_visibilities = &tcx.effective_visibilities(());
        let mut missing = MissingStabilityAnnotations { tcx, effective_visibilities };
        if module_def_id.is_top_level_module() {
            missing.check_missing_stability(CRATE_DEF_ID);
        }
        tcx.hir_visit_item_likes_in_module(module_def_id, &mut missing);
    }

    if module_def_id.is_top_level_module() {
        check_unused_or_stable_features(tcx)
    }
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers {
        check_mod_unstable_api_usage,
        stability_implications,
        lookup_stability,
        lookup_const_stability,
        lookup_default_body_stability,
        lookup_deprecation_entry,
        ..*providers
    };
}

struct Checker<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> Visitor<'tcx> for Checker<'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    /// Because stability levels are scoped lexically, we want to walk
    /// nested items in the context of the outer item, so enable
    /// deep-walking.
    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }

    fn visit_item(&mut self, item: &'tcx hir::Item<'tcx>) {
        match item.kind {
            hir::ItemKind::ExternCrate(_, ident) => {
                // compiler-generated `extern crate` items have a dummy span.
                // `std` is still checked for the `restricted-std` feature.
                if item.span.is_dummy() && ident.name != sym::std {
                    return;
                }

                let Some(cnum) = self.tcx.extern_mod_stmt_cnum(item.owner_id.def_id) else {
                    return;
                };
                let def_id = cnum.as_def_id();
                self.tcx.check_stability(def_id, Some(item.hir_id()), item.span, None);
            }

            // For implementations of traits, check the stability of each item
            // individually as it's possible to have a stable trait with unstable
            // items.
            hir::ItemKind::Impl(hir::Impl { of_trait: Some(of_trait), self_ty, items, .. }) => {
                let features = self.tcx.features();
                if features.staged_api() {
                    let attrs = self.tcx.hir_attrs(item.hir_id());
                    let stab = find_attr!(attrs, AttributeKind::Stability{stability, span} => (*stability, *span));

                    // FIXME(jdonszelmann): make it impossible to miss the or_else in the typesystem
                    let const_stab = find_attr!(attrs, AttributeKind::ConstStability{stability, ..} => *stability);

                    let unstable_feature_stab =
                        find_attr!(attrs, AttributeKind::UnstableFeatureBound(i) => i)
                            .map(|i| i.as_slice())
                            .unwrap_or_default();

                    // If this impl block has an #[unstable] attribute, give an
                    // error if all involved types and traits are stable, because
                    // it will have no effect.
                    // See: https://github.com/rust-lang/rust/issues/55436
                    //
                    // The exception is when there are both  #[unstable_feature_bound(..)] and
                    //  #![unstable(feature = "..", issue = "..")] that have the same symbol because
                    // that can effectively mark an impl as unstable.
                    //
                    // For example:
                    // ```
                    // #[unstable_feature_bound(feat_foo)]
                    // #[unstable(feature = "feat_foo", issue = "none")]
                    // impl Foo for Bar {}
                    // ```
                    if let Some((
                        Stability { level: StabilityLevel::Unstable { .. }, feature },
                        span,
                    )) = stab
                    {
                        let mut c = CheckTraitImplStable { tcx: self.tcx, fully_stable: true };
                        c.visit_ty_unambig(self_ty);
                        c.visit_trait_ref(&of_trait.trait_ref);

                        // Skip the lint if the impl is marked as unstable using
                        // #[unstable_feature_bound(..)]
                        let mut unstable_feature_bound_in_effect = false;
                        for (unstable_bound_feat_name, _) in unstable_feature_stab {
                            if *unstable_bound_feat_name == feature {
                                unstable_feature_bound_in_effect = true;
                            }
                        }

                        // do not lint when the trait isn't resolved, since resolution error should
                        // be fixed first
                        if of_trait.trait_ref.path.res != Res::Err
                            && c.fully_stable
                            && !unstable_feature_bound_in_effect
                        {
                            self.tcx.emit_node_span_lint(
                                INEFFECTIVE_UNSTABLE_TRAIT_IMPL,
                                item.hir_id(),
                                span,
                                errors::IneffectiveUnstableImpl,
                            );
                        }
                    }

                    if features.const_trait_impl()
                        && let hir::Constness::Const = of_trait.constness
                    {
                        let stable_or_implied_stable = match const_stab {
                            None => true,
                            Some(stab) if stab.is_const_stable() => {
                                // `#![feature(const_trait_impl)]` is unstable, so any impl declared stable
                                // needs to have an error emitted.
                                // Note: Remove this error once `const_trait_impl` is stabilized
                                self.tcx
                                    .dcx()
                                    .emit_err(errors::TraitImplConstStable { span: item.span });
                                true
                            }
                            Some(_) => false,
                        };

                        if let Some(trait_id) = of_trait.trait_ref.trait_def_id()
                            && let Some(const_stab) = self.tcx.lookup_const_stability(trait_id)
                        {
                            // the const stability of a trait impl must match the const stability on the trait.
                            if const_stab.is_const_stable() != stable_or_implied_stable {
                                let trait_span = self.tcx.def_ident_span(trait_id).unwrap();

                                let impl_stability = if stable_or_implied_stable {
                                    errors::ImplConstStability::Stable { span: item.span }
                                } else {
                                    errors::ImplConstStability::Unstable { span: item.span }
                                };
                                let trait_stability = if const_stab.is_const_stable() {
                                    errors::TraitConstStability::Stable { span: trait_span }
                                } else {
                                    errors::TraitConstStability::Unstable { span: trait_span }
                                };

                                self.tcx.dcx().emit_err(errors::TraitImplConstStabilityMismatch {
                                    span: item.span,
                                    impl_stability,
                                    trait_stability,
                                });
                            }
                        }
                    }
                }

                if let hir::Constness::Const = of_trait.constness
                    && let Some(def_id) = of_trait.trait_ref.trait_def_id()
                {
                    // FIXME(const_trait_impl): Improve the span here.
                    self.tcx.check_const_stability(
                        def_id,
                        of_trait.trait_ref.path.span,
                        of_trait.trait_ref.path.span,
                    );
                }

                for impl_item_ref in items {
                    let impl_item = self.tcx.associated_item(impl_item_ref.owner_id);

                    if let AssocContainer::TraitImpl(Ok(def_id)) = impl_item.container {
                        // Pass `None` to skip deprecation warnings.
                        self.tcx.check_stability(
                            def_id,
                            None,
                            self.tcx.def_span(impl_item_ref.owner_id),
                            None,
                        );
                    }
                }
            }

            _ => (/* pass */),
        }
        intravisit::walk_item(self, item);
    }

    fn visit_poly_trait_ref(&mut self, t: &'tcx hir::PolyTraitRef<'tcx>) {
        match t.modifiers.constness {
            hir::BoundConstness::Always(span) | hir::BoundConstness::Maybe(span) => {
                if let Some(def_id) = t.trait_ref.trait_def_id() {
                    self.tcx.check_const_stability(def_id, t.trait_ref.path.span, span);
                }
            }
            hir::BoundConstness::Never => {}
        }
        intravisit::walk_poly_trait_ref(self, t);
    }

    fn visit_path(&mut self, path: &hir::Path<'tcx>, id: hir::HirId) {
        if let Some(def_id) = path.res.opt_def_id() {
            let method_span = path.segments.last().map(|s| s.ident.span);
            let item_is_allowed = self.tcx.check_stability_allow_unstable(
                def_id,
                Some(id),
                path.span,
                method_span,
                if is_unstable_reexport(self.tcx, id) {
                    AllowUnstable::Yes
                } else {
                    AllowUnstable::No
                },
            );

            if item_is_allowed {
                // The item itself is allowed; check whether the path there is also allowed.
                let is_allowed_through_unstable_modules: Option<Symbol> =
                    self.tcx.lookup_stability(def_id).and_then(|stab| match stab.level {
                        StabilityLevel::Stable { allowed_through_unstable_modules, .. } => {
                            allowed_through_unstable_modules
                        }
                        _ => None,
                    });

                // Check parent modules stability as well if the item the path refers to is itself
                // stable. We only emit errors for unstable path segments if the item is stable
                // or allowed because stability is often inherited, so the most common case is that
                // both the segments and the item are unstable behind the same feature flag.
                //
                // We check here rather than in `visit_path_segment` to prevent visiting the last
                // path segment twice
                //
                // We include special cases via #[rustc_allowed_through_unstable_modules] for items
                // that were accidentally stabilized through unstable paths before this check was
                // added, such as `core::intrinsics::transmute`
                let parents = path.segments.iter().rev().skip(1);
                for path_segment in parents {
                    if let Some(def_id) = path_segment.res.opt_def_id() {
                        match is_allowed_through_unstable_modules {
                            None => {
                                // Emit a hard stability error if this path is not stable.

                                // use `None` for id to prevent deprecation check
                                self.tcx.check_stability_allow_unstable(
                                    def_id,
                                    None,
                                    path.span,
                                    None,
                                    if is_unstable_reexport(self.tcx, id) {
                                        AllowUnstable::Yes
                                    } else {
                                        AllowUnstable::No
                                    },
                                );
                            }
                            Some(deprecation) => {
                                // Call the stability check directly so that we can control which
                                // diagnostic is emitted.
                                let eval_result = self.tcx.eval_stability_allow_unstable(
                                    def_id,
                                    None,
                                    path.span,
                                    None,
                                    if is_unstable_reexport(self.tcx, id) {
                                        AllowUnstable::Yes
                                    } else {
                                        AllowUnstable::No
                                    },
                                );
                                let is_allowed = matches!(eval_result, EvalResult::Allow);
                                if !is_allowed {
                                    // Calculating message for lint involves calling `self.def_path_str`,
                                    // which will by default invoke the expensive `visible_parent_map` query.
                                    // Skip all that work if the lint is allowed anyway.
                                    if self.tcx.lint_level_at_node(DEPRECATED, id).level
                                        == lint::Level::Allow
                                    {
                                        return;
                                    }
                                    // Show a deprecation message.
                                    let def_path =
                                        with_no_trimmed_paths!(self.tcx.def_path_str(def_id));
                                    let def_kind = self.tcx.def_descr(def_id);
                                    let diag = Deprecated {
                                        sub: None,
                                        kind: def_kind.to_owned(),
                                        path: def_path,
                                        note: Some(deprecation),
                                        since_kind: lint::DeprecatedSinceKind::InEffect,
                                    };
                                    self.tcx.emit_node_span_lint(
                                        DEPRECATED,
                                        id,
                                        method_span.unwrap_or(path.span),
                                        diag,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        intravisit::walk_path(self, path)
    }
}

/// Check whether a path is a `use` item that has been marked as unstable.
///
/// See issue #94972 for details on why this is a special case
fn is_unstable_reexport(tcx: TyCtxt<'_>, id: hir::HirId) -> bool {
    // Get the LocalDefId so we can lookup the item to check the kind.
    let Some(owner) = id.as_owner() else {
        return false;
    };
    let def_id = owner.def_id;

    let Some(stab) = tcx.lookup_stability(def_id) else {
        return false;
    };

    if stab.level.is_stable() {
        // The re-export is not marked as unstable, don't override
        return false;
    }

    // If this is a path that isn't a use, we don't need to do anything special
    if !matches!(tcx.hir_expect_item(def_id).kind, ItemKind::Use(..)) {
        return false;
    }

    true
}

struct CheckTraitImplStable<'tcx> {
    tcx: TyCtxt<'tcx>,
    fully_stable: bool,
}

impl<'tcx> Visitor<'tcx> for CheckTraitImplStable<'tcx> {
    fn visit_path(&mut self, path: &hir::Path<'tcx>, _id: hir::HirId) {
        if let Some(def_id) = path.res.opt_def_id()
            && let Some(stab) = self.tcx.lookup_stability(def_id)
        {
            self.fully_stable &= stab.level.is_stable();
        }
        intravisit::walk_path(self, path)
    }

    fn visit_trait_ref(&mut self, t: &'tcx TraitRef<'tcx>) {
        if let Res::Def(DefKind::Trait, trait_did) = t.path.res {
            if let Some(stab) = self.tcx.lookup_stability(trait_did) {
                self.fully_stable &= stab.level.is_stable();
            }
        }
        intravisit::walk_trait_ref(self, t)
    }

    fn visit_ty(&mut self, t: &'tcx Ty<'tcx, AmbigArg>) {
        if let TyKind::Never = t.kind {
            self.fully_stable = false;
        }
        if let TyKind::FnPtr(function) = t.kind {
            if extern_abi_stability(function.abi).is_err() {
                self.fully_stable = false;
            }
        }
        intravisit::walk_ty(self, t)
    }

    fn visit_fn_decl(&mut self, fd: &'tcx hir::FnDecl<'tcx>) {
        for ty in fd.inputs {
            self.visit_ty_unambig(ty)
        }
        if let hir::FnRetTy::Return(output_ty) = fd.output {
            match output_ty.kind {
                TyKind::Never => {} // `-> !` is stable
                _ => self.visit_ty_unambig(output_ty),
            }
        }
    }
}

/// Given the list of enabled features that were not language features (i.e., that
/// were expected to be library features), and the list of features used from
/// libraries, identify activated features that don't exist and error about them.
// This is `pub` for rustdoc. rustc should call it through `check_mod_unstable_api_usage`.
pub fn check_unused_or_stable_features(tcx: TyCtxt<'_>) {
    let _prof_timer = tcx.sess.timer("unused_lib_feature_checking");

    let enabled_lang_features = tcx.features().enabled_lang_features();
    let mut lang_features = UnordSet::default();
    for EnabledLangFeature { gate_name, attr_sp, stable_since } in enabled_lang_features {
        if let Some(version) = stable_since {
            // Warn if the user has enabled an already-stable lang feature.
            unnecessary_stable_feature_lint(tcx, *attr_sp, *gate_name, *version);
        }
        if !lang_features.insert(gate_name) {
            // Warn if the user enables a lang feature multiple times.
            tcx.dcx().emit_err(errors::DuplicateFeatureErr { span: *attr_sp, feature: *gate_name });
        }
    }

    let enabled_lib_features = tcx.features().enabled_lib_features();
    let mut remaining_lib_features = FxIndexMap::default();
    for EnabledLibFeature { gate_name, attr_sp } in enabled_lib_features {
        if remaining_lib_features.contains_key(gate_name) {
            // Warn if the user enables a lib feature multiple times.
            tcx.dcx().emit_err(errors::DuplicateFeatureErr { span: *attr_sp, feature: *gate_name });
        }
        remaining_lib_features.insert(*gate_name, *attr_sp);
    }
    // `stdbuild` has special handling for `libc`, so we need to
    // recognise the feature when building std.
    // Likewise, libtest is handled specially, so `test` isn't
    // available as we'd like it to be.
    // FIXME: only remove `libc` when `stdbuild` is enabled.
    // FIXME: remove special casing for `test`.
    // FIXME(#120456) - is `swap_remove` correct?
    remaining_lib_features.swap_remove(&sym::libc);
    remaining_lib_features.swap_remove(&sym::test);

    /// For each feature in `defined_features`..
    ///
    /// - If it is in `remaining_lib_features` (those features with `#![feature(..)]` attributes in
    ///   the current crate), check if it is stable (or partially stable) and thus an unnecessary
    ///   attribute.
    /// - If it is in `remaining_implications` (a feature that is referenced by an `implied_by`
    ///   from the current crate), then remove it from the remaining implications.
    ///
    /// Once this function has been invoked for every feature (local crate and all extern crates),
    /// then..
    ///
    /// - If features remain in `remaining_lib_features`, then the user has enabled a feature that
    ///   does not exist.
    /// - If features remain in `remaining_implications`, the `implied_by` refers to a feature that
    ///   does not exist.
    ///
    /// By structuring the code in this way: checking the features defined from each crate one at a
    /// time, less loading from metadata is performed and thus compiler performance is improved.
    fn check_features<'tcx>(
        tcx: TyCtxt<'tcx>,
        remaining_lib_features: &mut FxIndexMap<Symbol, Span>,
        remaining_implications: &mut UnordMap<Symbol, Symbol>,
        defined_features: &LibFeatures,
        all_implications: &UnordMap<Symbol, Symbol>,
    ) {
        for (feature, stability) in defined_features.to_sorted_vec() {
            if let FeatureStability::AcceptedSince(since) = stability
                && let Some(span) = remaining_lib_features.get(&feature)
            {
                // Warn if the user has enabled an already-stable lib feature.
                if let Some(implies) = all_implications.get(&feature) {
                    unnecessary_partially_stable_feature_lint(tcx, *span, feature, *implies, since);
                } else {
                    unnecessary_stable_feature_lint(tcx, *span, feature, since);
                }
            }
            // FIXME(#120456) - is `swap_remove` correct?
            remaining_lib_features.swap_remove(&feature);

            // `feature` is the feature doing the implying, but `implied_by` is the feature with
            // the attribute that establishes this relationship. `implied_by` is guaranteed to be a
            // feature defined in the local crate because `remaining_implications` is only the
            // implications from this crate.
            remaining_implications.remove(&feature);

            if let FeatureStability::Unstable { old_name: Some(alias) } = stability
                && let Some(span) = remaining_lib_features.swap_remove(&alias)
            {
                tcx.dcx().emit_err(errors::RenamedFeature { span, feature, alias });
            }

            if remaining_lib_features.is_empty() && remaining_implications.is_empty() {
                break;
            }
        }
    }

    // All local crate implications need to have the feature that implies it confirmed to exist.
    let mut remaining_implications = tcx.stability_implications(LOCAL_CRATE).clone();

    // We always collect the lib features enabled in the current crate, even if there are
    // no unknown features, because the collection also does feature attribute validation.
    let local_defined_features = tcx.lib_features(LOCAL_CRATE);
    if !remaining_lib_features.is_empty() || !remaining_implications.is_empty() {
        // Loading the implications of all crates is unavoidable to be able to emit the partial
        // stabilization diagnostic, but it can be avoided when there are no
        // `remaining_lib_features`.
        let mut all_implications = remaining_implications.clone();
        for &cnum in tcx.crates(()) {
            all_implications
                .extend_unord(tcx.stability_implications(cnum).items().map(|(k, v)| (*k, *v)));
        }

        check_features(
            tcx,
            &mut remaining_lib_features,
            &mut remaining_implications,
            local_defined_features,
            &all_implications,
        );

        for &cnum in tcx.crates(()) {
            if remaining_lib_features.is_empty() && remaining_implications.is_empty() {
                break;
            }
            check_features(
                tcx,
                &mut remaining_lib_features,
                &mut remaining_implications,
                tcx.lib_features(cnum),
                &all_implications,
            );
        }
    }

    for (feature, span) in remaining_lib_features {
        tcx.dcx().emit_err(errors::UnknownFeature { span, feature });
    }

    for (&implied_by, &feature) in remaining_implications.to_sorted_stable_ord() {
        let local_defined_features = tcx.lib_features(LOCAL_CRATE);
        let span = local_defined_features
            .stability
            .get(&feature)
            .expect("feature that implied another does not exist")
            .1;
        tcx.dcx().emit_err(errors::ImpliedFeatureNotExist { span, feature, implied_by });
    }

    // FIXME(#44232): the `used_features` table no longer exists, so we
    // don't lint about unused features. We should re-enable this one day!
}

fn unnecessary_partially_stable_feature_lint(
    tcx: TyCtxt<'_>,
    span: Span,
    feature: Symbol,
    implies: Symbol,
    since: Symbol,
) {
    tcx.emit_node_span_lint(
        lint::builtin::STABLE_FEATURES,
        hir::CRATE_HIR_ID,
        span,
        errors::UnnecessaryPartialStableFeature {
            span,
            line: tcx.sess.source_map().span_extend_to_line(span),
            feature,
            since,
            implies,
        },
    );
}

fn unnecessary_stable_feature_lint(
    tcx: TyCtxt<'_>,
    span: Span,
    feature: Symbol,
    mut since: Symbol,
) {
    if since.as_str() == VERSION_PLACEHOLDER {
        since = sym::env_CFG_RELEASE;
    }
    tcx.emit_node_span_lint(
        lint::builtin::STABLE_FEATURES,
        hir::CRATE_HIR_ID,
        span,
        errors::UnnecessaryStableFeature { feature, since },
    );
}
