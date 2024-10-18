//! A pass that annotates every item and method with its stability level,
//! propagating default levels lexically from parent to children ast nodes.

use std::num::NonZero;

use rustc_ast::NodeId;
use rustc_attr::{
    self as attr, ConstStability, DefaultBodyStability, DeprecatedSince, Deprecation, Stability,
};
use rustc_data_structures::unord::UnordMap;
use rustc_errors::{Diag, EmissionGuarantee};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId, LocalDefIdMap};
use rustc_hir::{self as hir, HirId};
use rustc_macros::{Decodable, Encodable, HashStable, Subdiagnostic};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_session::Session;
use rustc_session::errors::NightlyFeatureDiagnostic;
use rustc_session::lint::builtin::{DEPRECATED, DEPRECATED_IN_FUTURE, SOFT_UNSTABLE};
use rustc_session::lint::{BuiltinLintDiag, DeprecatedSinceKind, Level, Lint, LintBuffer};
use rustc_span::Span;
use rustc_span::symbol::{Symbol, sym};
use tracing::debug;

pub use self::StabilityLevel::*;
use crate::error::{
    SoftUnstableLibraryFeature, UnstableLibraryFeatureError, UnstableLibraryFeatureInfo,
    UnstableLibraryFeatureNote, UnstableLibraryFeatureSugg,
};
use crate::ty::{self, TyCtxt};

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum StabilityLevel {
    Unstable,
    Stable,
}

/// An entry in the `depr_map`.
#[derive(Copy, Clone, HashStable, Debug, Encodable, Decodable)]
pub struct DeprecationEntry {
    /// The metadata of the attribute associated with this entry.
    pub attr: Deprecation,
    /// The `DefId` where the attr was originally attached. `None` for non-local
    /// `DefId`'s.
    origin: Option<LocalDefId>,
}

impl DeprecationEntry {
    pub fn local(attr: Deprecation, def_id: LocalDefId) -> DeprecationEntry {
        DeprecationEntry { attr, origin: Some(def_id) }
    }

    pub fn external(attr: Deprecation) -> DeprecationEntry {
        DeprecationEntry { attr, origin: None }
    }

    pub fn same_origin(&self, other: &DeprecationEntry) -> bool {
        match (self.origin, other.origin) {
            (Some(o1), Some(o2)) => o1 == o2,
            _ => false,
        }
    }
}

/// A stability index, giving the stability level for items and methods.
#[derive(HashStable, Debug)]
pub struct Index<'tcx> {
    /// This is mostly a cache, except the stabilities of local items
    /// are filled by the annotator.
    pub stab_map: LocalDefIdMap<&'tcx Stability>,
    pub const_stab_map: LocalDefIdMap<&'tcx ConstStability>,
    pub default_body_stab_map: LocalDefIdMap<&'tcx DefaultBodyStability>,
    pub depr_map: LocalDefIdMap<DeprecationEntry>,
    /// Mapping from feature name to feature name based on the `implied_by` field of `#[unstable]`
    /// attributes. If a `#[unstable(feature = "implier", implied_by = "impliee")]` attribute
    /// exists, then this map will have a `impliee -> implier` entry.
    ///
    /// This mapping is necessary unless both the `#[stable]` and `#[unstable]` attributes should
    /// specify their implications (both `implies` and `implied_by`). If only one of the two
    /// attributes do (as in the current implementation, `implied_by` in `#[unstable]`), then this
    /// mapping is necessary for diagnostics. When a "unnecessary feature attribute" error is
    /// reported, only the `#[stable]` attribute information is available, so the map is necessary
    /// to know that the feature implies another feature. If it were reversed, and the `#[stable]`
    /// attribute had an `implies` meta item, then a map would be necessary when avoiding a "use of
    /// unstable feature" error for a feature that was implied.
    pub implications: UnordMap<Symbol, Symbol>,
}

impl<'tcx> Index<'tcx> {
    pub fn local_stability(&self, def_id: LocalDefId) -> Option<&'tcx Stability> {
        self.stab_map.get(&def_id).copied()
    }

    pub fn local_const_stability(&self, def_id: LocalDefId) -> Option<&'tcx ConstStability> {
        self.const_stab_map.get(&def_id).copied()
    }

    pub fn local_default_body_stability(
        &self,
        def_id: LocalDefId,
    ) -> Option<&'tcx DefaultBodyStability> {
        self.default_body_stab_map.get(&def_id).copied()
    }

    pub fn local_deprecation_entry(&self, def_id: LocalDefId) -> Option<DeprecationEntry> {
        self.depr_map.get(&def_id).cloned()
    }
}

/// Produces notes with reasons and issue numbers for missing unstable library features
pub fn unstable_notes(
    denials: &[EvalDenial],
) -> (UnstableLibraryFeatureNote, Vec<UnstableLibraryFeatureInfo>) {
    use UnstableLibraryFeatureInfo::*;

    let count = denials.len();
    let features = denials.iter().map(|d| d.feature).collect();
    // if there's only one missing feature, put the reason for it on the same line
    let (single_feature_has_reason, reason_for_single_feature) = match denials {
        [EvalDenial { reason: Some(reason), .. }] => (true, reason.to_string()),
        _ => (false, String::new()),
    };
    let primary = UnstableLibraryFeatureNote {
        features,
        count,
        single_feature_has_reason,
        reason_for_single_feature,
    };

    let info = denials.iter().flat_map(|&EvalDenial { feature, reason, issue }| {
        // don't repeat the reason if it's on the first line
        let reason_info = if !single_feature_has_reason && let Some(reason) = reason {
            Some(Reason { reason: reason.to_string(), feature })
        } else {
            None
        };
        // don't list the feature with the issue if there's only one feature or if it was just
        // provided by a reason note
        let show_feature = count > 1 && reason_info.is_none();
        let issue_info = issue.map(|issue| Issue { issue, feature, show_feature });

        reason_info.into_iter().chain(issue_info)
    });

    (primary, info.collect())
}

/// Produces help/suggestions/notes for how to enable missing unstable features on nightly
pub fn unstable_nightly_subdiags(
    sess: &Session,
    denials: &[EvalDenial],
    inject_span: Option<Span>,
) -> Vec<NightlyFeatureDiagnostic> {
    if sess.is_nightly_build() {
        let feature = denials.iter().map(|d| d.feature.as_str()).intersperse(", ").collect();

        std::iter::once(if let Some(span) = inject_span {
            NightlyFeatureDiagnostic::Suggestion { feature, span }
        } else {
            NightlyFeatureDiagnostic::Help { feature }
        })
        .chain(NightlyFeatureDiagnostic::suggest_upgrade_compiler(sess))
        .collect()
    } else {
        vec![]
    }
}

/// Emits an error for the use of disabled unstable library features
pub fn report_unstable(
    sess: &Session,
    denials: &[EvalDenial],
    suggestions: Vec<UnstableLibraryFeatureSugg>,
    span: Span,
) {
    let (message, info) = unstable_notes(denials);
    let UnstableLibraryFeatureNote {
        features,
        count,
        single_feature_has_reason,
        reason_for_single_feature,
    } = message;
    let nightly_subdiags = unstable_nightly_subdiags(sess, denials, None);

    sess.dcx().emit_err(UnstableLibraryFeatureError {
        span,
        features,
        count,
        single_feature_has_reason,
        reason_for_single_feature,
        info,
        suggestions,
        nightly_subdiags,
    });
}

/// Constructs a lint diagnostic for [`SOFT_UNSTABLE`]
pub fn soft_unstable(
    sess: &Session,
    denials: &[EvalDenial],
    suggestions: Vec<UnstableLibraryFeatureSugg>,
) -> SoftUnstableLibraryFeature {
    let (message, mut info) = unstable_notes(denials);
    let UnstableLibraryFeatureNote {
        features,
        count,
        single_feature_has_reason,
        reason_for_single_feature,
    } = message;
    let nightly_subdiags = unstable_nightly_subdiags(sess, denials, None);

    // Since SOFT_UNSTABLE puts an issue link between the contents of `message` and `info`,
    // always provide the feature name for context if the first info is an issue link.
    if let Some(UnstableLibraryFeatureInfo::Issue { show_feature, .. }) = info.first_mut() {
        *show_feature = true;
    }

    SoftUnstableLibraryFeature {
        features,
        count,
        single_feature_has_reason,
        reason_for_single_feature,
        info,
        suggestions,
        nightly_subdiags,
    }
}

/// Buffers a [`SOFT_UNSTABLE`] lint for soft-unstable macro usage
pub fn report_soft_unstable_macro(
    lint_buffer: &mut LintBuffer,
    denials: &[EvalDenial],
    span: Span,
    node_id: NodeId,
) {
    lint_buffer.buffer_lint(SOFT_UNSTABLE, node_id, span, BuiltinLintDiag::SoftUnstableMacro {
        features: denials
            .iter()
            .map(|&EvalDenial { feature, reason, issue }| (feature, reason, issue))
            .collect(),
    });
}

fn deprecation_lint(is_in_effect: bool) -> &'static Lint {
    if is_in_effect { DEPRECATED } else { DEPRECATED_IN_FUTURE }
}

#[derive(Subdiagnostic)]
#[suggestion(
    middle_deprecated_suggestion,
    code = "{suggestion}",
    style = "verbose",
    applicability = "machine-applicable"
)]
pub struct DeprecationSuggestion {
    #[primary_span]
    pub span: Span,

    pub kind: String,
    pub suggestion: Symbol,
}

pub struct Deprecated {
    pub sub: Option<DeprecationSuggestion>,

    // FIXME: make this translatable
    pub kind: String,
    pub path: String,
    pub note: Option<Symbol>,
    pub since_kind: DeprecatedSinceKind,
}

impl<'a, G: EmissionGuarantee> rustc_errors::LintDiagnostic<'a, G> for Deprecated {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, G>) {
        diag.primary_message(match &self.since_kind {
            DeprecatedSinceKind::InEffect => crate::fluent_generated::middle_deprecated,
            DeprecatedSinceKind::InFuture => crate::fluent_generated::middle_deprecated_in_future,
            DeprecatedSinceKind::InVersion(_) => {
                crate::fluent_generated::middle_deprecated_in_version
            }
        });
        diag.arg("kind", self.kind);
        diag.arg("path", self.path);
        if let DeprecatedSinceKind::InVersion(version) = self.since_kind {
            diag.arg("version", version);
        }
        if let Some(note) = self.note {
            diag.arg("has_note", true);
            diag.arg("note", note);
        } else {
            diag.arg("has_note", false);
        }
        if let Some(sub) = self.sub {
            diag.subdiagnostic(sub);
        }
    }
}

fn deprecated_since_kind(is_in_effect: bool, since: DeprecatedSince) -> DeprecatedSinceKind {
    if is_in_effect {
        DeprecatedSinceKind::InEffect
    } else {
        match since {
            DeprecatedSince::RustcVersion(version) => {
                DeprecatedSinceKind::InVersion(version.to_string())
            }
            DeprecatedSince::Future => DeprecatedSinceKind::InFuture,
            DeprecatedSince::NonStandard(_)
            | DeprecatedSince::Unspecified
            | DeprecatedSince::Err => {
                unreachable!("this deprecation is always in effect; {since:?}")
            }
        }
    }
}

pub fn early_report_macro_deprecation(
    lint_buffer: &mut LintBuffer,
    depr: &Deprecation,
    span: Span,
    node_id: NodeId,
    path: String,
) {
    if span.in_derive_expansion() {
        return;
    }

    let is_in_effect = depr.is_in_effect();
    let diag = BuiltinLintDiag::DeprecatedMacro {
        suggestion: depr.suggestion,
        suggestion_span: span,
        note: depr.note,
        path,
        since_kind: deprecated_since_kind(is_in_effect, depr.since),
    };
    lint_buffer.buffer_lint(deprecation_lint(is_in_effect), node_id, span, diag);
}

fn late_report_deprecation(
    tcx: TyCtxt<'_>,
    depr: &Deprecation,
    span: Span,
    method_span: Option<Span>,
    hir_id: HirId,
    def_id: DefId,
) {
    if span.in_derive_expansion() {
        return;
    }

    let def_path = with_no_trimmed_paths!(tcx.def_path_str(def_id));
    let def_kind = tcx.def_descr(def_id);
    let is_in_effect = depr.is_in_effect();

    let method_span = method_span.unwrap_or(span);
    let suggestion =
        if let hir::Node::Expr(_) = tcx.hir_node(hir_id) { depr.suggestion } else { None };
    let diag = Deprecated {
        sub: suggestion.map(|suggestion| DeprecationSuggestion {
            span: method_span,
            kind: def_kind.to_owned(),
            suggestion,
        }),
        kind: def_kind.to_owned(),
        path: def_path,
        note: depr.note,
        since_kind: deprecated_since_kind(is_in_effect, depr.since),
    };
    tcx.emit_node_span_lint(deprecation_lint(is_in_effect), hir_id, method_span, diag);
}

/// Result of `TyCtxt::eval_stability`.
pub enum EvalResult {
    /// We can use the item because it is stable or we enabled the
    /// corresponding feature gates.
    Allow,
    /// We cannot use the item because it is unstable and we did not enable the
    /// corresponding feature gates.
    Deny { denials: Vec<EvalDenial>, suggestions: Vec<UnstableLibraryFeatureSugg>, is_soft: bool },
    /// The item does not have the `#[stable]` or `#[unstable]` marker assigned.
    Unmarked,
}

/// An instance of a disabled feature required for an unstable item
#[derive(Debug)]
pub struct EvalDenial {
    pub feature: Symbol,
    pub reason: Option<Symbol>,
    pub issue: Option<NonZero<u32>>,
}

impl<'a> From<&'a attr::Unstability> for EvalDenial {
    fn from(unstab: &attr::Unstability) -> Self {
        let reason = unstab.reason.to_opt_reason();
        EvalDenial { feature: unstab.feature, reason, issue: unstab.issue }
    }
}

// See issue #38412.
fn skip_stability_check_due_to_privacy(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    if tcx.def_kind(def_id) == DefKind::TyParam {
        // Have no visibility, considered public for the purpose of this check.
        return false;
    }
    match tcx.visibility(def_id) {
        // Must check stability for `pub` items.
        ty::Visibility::Public => false,

        // These are not visible outside crate; therefore
        // stability markers are irrelevant, if even present.
        ty::Visibility::Restricted(..) => true,
    }
}

// See issue #83250.
fn suggestion_for_allocator_api(
    tcx: TyCtxt<'_>,
    def_id: DefId,
    span: Span,
    feature: Symbol,
) -> Option<UnstableLibraryFeatureSugg> {
    if feature == sym::allocator_api {
        if let Some(trait_) = tcx.opt_parent(def_id) {
            if tcx.is_diagnostic_item(sym::Vec, trait_) {
                let sm = tcx.sess.psess.source_map();
                let inner_types = sm.span_extend_to_prev_char(span, '<', true);
                if let Ok(snippet) = sm.span_to_snippet(inner_types) {
                    return Some(UnstableLibraryFeatureSugg::ForAllocatorApi {
                        inner_types,
                        snippet,
                    });
                }
            }
        }
    }
    None
}

/// An override option for eval_stability.
pub enum AllowUnstable {
    /// Don't emit an unstable error for the item
    Yes,
    /// Handle the item normally
    No,
}

impl<'tcx> TyCtxt<'tcx> {
    /// Evaluates the stability of an item.
    ///
    /// Returns `EvalResult::Allow` if the item is stable, or unstable but the corresponding
    /// `#![feature]` has been provided. Returns `EvalResult::Deny` which describes the offending
    /// unstable feature otherwise.
    ///
    /// If `id` is `Some(_)`, this function will also check if the item at `def_id` has been
    /// deprecated. If the item is indeed deprecated, we will emit a deprecation lint attached to
    /// `id`.
    pub fn eval_stability(
        self,
        def_id: DefId,
        id: Option<HirId>,
        span: Span,
        method_span: Option<Span>,
    ) -> EvalResult {
        self.eval_stability_allow_unstable(def_id, id, span, method_span, AllowUnstable::No)
    }

    /// Evaluates the stability of an item.
    ///
    /// Returns `EvalResult::Allow` if the item is stable, or unstable but the corresponding
    /// `#![feature]`s have been provided. Returns `EvalResult::Deny` which describes the offending
    /// unstable features otherwise.
    ///
    /// If `id` is `Some(_)`, this function will also check if the item at `def_id` has been
    /// deprecated. If the item is indeed deprecated, we will emit a deprecation lint attached to
    /// `id`.
    ///
    /// Pass `AllowUnstable::Yes` to `allow_unstable` to force an unstable item to be allowed. Deprecation warnings will be emitted normally.
    pub fn eval_stability_allow_unstable(
        self,
        def_id: DefId,
        id: Option<HirId>,
        span: Span,
        method_span: Option<Span>,
        allow_unstable: AllowUnstable,
    ) -> EvalResult {
        // Deprecated attributes apply in-crate and cross-crate.
        if let Some(id) = id {
            if let Some(depr_entry) = self.lookup_deprecation_entry(def_id) {
                let parent_def_id = self.hir().get_parent_item(id);
                let skip = self
                    .lookup_deprecation_entry(parent_def_id.to_def_id())
                    .is_some_and(|parent_depr| parent_depr.same_origin(&depr_entry));

                // #[deprecated] doesn't emit a notice if we're not on the
                // topmost deprecation. For example, if a struct is deprecated,
                // the use of a field won't be linted.
                //
                // With #![staged_api], we want to emit down the whole
                // hierarchy.
                let depr_attr = &depr_entry.attr;
                if !skip || depr_attr.is_since_rustc_version() {
                    // Calculating message for lint involves calling `self.def_path_str`.
                    // Which by default to calculate visible path will invoke expensive `visible_parent_map` query.
                    // So we skip message calculation altogether, if lint is allowed.
                    let lint = deprecation_lint(depr_attr.is_in_effect());
                    if self.lint_level_at_node(lint, id).0 != Level::Allow {
                        late_report_deprecation(self, depr_attr, span, method_span, id, def_id);
                    }
                }
            };
        }

        let is_staged_api = self.lookup_stability(def_id.krate.as_def_id()).is_some();
        if !is_staged_api {
            return EvalResult::Allow;
        }

        // Only the cross-crate scenario matters when checking unstable APIs
        let cross_crate = !def_id.is_local();
        if !cross_crate {
            return EvalResult::Allow;
        }

        let stability = self.lookup_stability(def_id);
        debug!(
            "stability: \
                inspecting def_id={:?} span={:?} of stability={:?}",
            def_id, span, stability
        );

        // Issue #38412: private items lack stability markers.
        if skip_stability_check_due_to_privacy(self, def_id) {
            return EvalResult::Allow;
        }

        match stability {
            Some(Stability { level: attr::Unstable { unstables, is_soft } }) => {
                if matches!(allow_unstable, AllowUnstable::Yes) {
                    return EvalResult::Allow;
                }

                let mut denials = vec![];
                let mut suggestions = vec![];

                for unstability in unstables {
                    let &attr::Unstability { feature, issue, .. } = unstability;
                    if span.allows_unstable(feature) {
                        debug!("stability: skipping span={:?} since it is internal", span);
                        continue;
                    }
                    if self.features().declared(feature) {
                        continue;
                    }

                    // If this item was previously part of a now-stabilized feature which is still
                    // active (i.e. the user hasn't removed the attribute for the stabilized feature
                    // yet) then allow use of this item.
                    if let Some(implied_by) = unstability.implied_by
                        && self.features().declared(implied_by)
                    {
                        continue;
                    }

                    // When we're compiling the compiler itself we may pull in
                    // crates from crates.io, but those crates may depend on other
                    // crates also pulled in from crates.io. We want to ideally be
                    // able to compile everything without requiring upstream
                    // modifications, so in the case that this looks like a
                    // `rustc_private` crate (e.g., a compiler crate) and we also have
                    // the `-Z force-unstable-if-unmarked` flag present (we're
                    // compiling a compiler crate), then let this missing feature
                    // annotation slide.
                    if feature == sym::rustc_private
                        && issue == NonZero::new(27812)
                        && self.sess.opts.unstable_opts.force_unstable_if_unmarked
                    {
                        continue;
                    }

                    denials.push(unstability.into());
                    suggestions.extend(suggestion_for_allocator_api(self, def_id, span, feature));
                }

                if denials.is_empty() {
                    EvalResult::Allow
                } else {
                    EvalResult::Deny { denials, suggestions, is_soft: *is_soft }
                }
            }
            Some(_) => {
                // Stable APIs are always ok to call and deprecated APIs are
                // handled by the lint emitting logic above.
                EvalResult::Allow
            }
            None => EvalResult::Unmarked,
        }
    }

    /// Evaluates the default-impl stability of an item.
    ///
    /// Returns `EvalResult::Allow` if the item's default implementation is stable, or unstable but the corresponding
    /// `#![feature]`s have been provided. Returns `EvalResult::Deny` which describes the offending
    /// unstable features otherwise.
    pub fn eval_default_body_stability(self, def_id: DefId, span: Span) -> EvalResult {
        let is_staged_api = self.lookup_stability(def_id.krate.as_def_id()).is_some();
        if !is_staged_api {
            return EvalResult::Allow;
        }

        // Only the cross-crate scenario matters when checking unstable APIs
        let cross_crate = !def_id.is_local();
        if !cross_crate {
            return EvalResult::Allow;
        }

        let stability = self.lookup_default_body_stability(def_id);
        debug!(
            "body stability: inspecting def_id={def_id:?} span={span:?} of stability={stability:?}"
        );

        // Issue #38412: private items lack stability markers.
        if skip_stability_check_due_to_privacy(self, def_id) {
            return EvalResult::Allow;
        }

        match stability {
            Some(DefaultBodyStability { level: attr::Unstable { unstables, is_soft } }) => {
                let mut denials = vec![];
                for unstability in unstables {
                    let feature = unstability.feature;
                    if span.allows_unstable(feature) {
                        debug!("body stability: skipping span={:?} since it is internal", span);
                        continue;
                    }
                    if self.features().declared(feature) {
                        continue;
                    }

                    denials.push(unstability.into());
                }

                if denials.is_empty() {
                    EvalResult::Allow
                } else {
                    EvalResult::Deny { denials, is_soft: *is_soft, suggestions: vec![] }
                }
            }
            Some(_) => {
                // Stable APIs are always ok to call
                EvalResult::Allow
            }
            None => EvalResult::Unmarked,
        }
    }

    /// Checks if an item is stable or error out.
    ///
    /// If the item defined by `def_id` is unstable and the corresponding `#![feature]` does not
    /// exist, emits an error.
    ///
    /// This function will also check if the item is deprecated.
    /// If so, and `id` is not `None`, a deprecated lint attached to `id` will be emitted.
    ///
    /// Returns `true` if item is allowed aka, stable or unstable under an enabled feature.
    pub fn check_stability(
        self,
        def_id: DefId,
        id: Option<HirId>,
        span: Span,
        method_span: Option<Span>,
    ) -> bool {
        self.check_stability_allow_unstable(def_id, id, span, method_span, AllowUnstable::No)
    }

    /// Checks if an item is stable or error out.
    ///
    /// If the item defined by `def_id` is unstable and the corresponding `#![feature]` does not
    /// exist, emits an error.
    ///
    /// This function will also check if the item is deprecated.
    /// If so, and `id` is not `None`, a deprecated lint attached to `id` will be emitted.
    ///
    /// Pass `AllowUnstable::Yes` to `allow_unstable` to force an unstable item to be allowed. Deprecation warnings will be emitted normally.
    ///
    /// Returns `true` if item is allowed aka, stable or unstable under an enabled feature.
    pub fn check_stability_allow_unstable(
        self,
        def_id: DefId,
        id: Option<HirId>,
        span: Span,
        method_span: Option<Span>,
        allow_unstable: AllowUnstable,
    ) -> bool {
        self.check_optional_stability(
            def_id,
            id,
            span,
            method_span,
            allow_unstable,
            |span, def_id| {
                // The API could be uncallable for other reasons, for example when a private module
                // was referenced.
                self.dcx().span_delayed_bug(span, format!("encountered unmarked API: {def_id:?}"));
            },
        )
    }

    /// Like `check_stability`, except that we permit items to have custom behaviour for
    /// missing stability attributes (not necessarily just emit a `bug!`). This is necessary
    /// for default generic parameters, which only have stability attributes if they were
    /// added after the type on which they're defined.
    ///
    /// Returns `true` if item is allowed aka, stable or unstable under an enabled feature.
    pub fn check_optional_stability(
        self,
        def_id: DefId,
        id: Option<HirId>,
        span: Span,
        method_span: Option<Span>,
        allow_unstable: AllowUnstable,
        unmarked: impl FnOnce(Span, DefId),
    ) -> bool {
        let eval_result =
            self.eval_stability_allow_unstable(def_id, id, span, method_span, allow_unstable);
        let is_allowed = matches!(eval_result, EvalResult::Allow);
        match eval_result {
            EvalResult::Allow => {}
            EvalResult::Deny { denials, is_soft, suggestions } => {
                if is_soft {
                    self.emit_node_span_lint(
                        SOFT_UNSTABLE,
                        id.unwrap_or(hir::CRATE_HIR_ID),
                        span,
                        soft_unstable(self.sess, &denials, suggestions),
                    );
                } else {
                    report_unstable(self.sess, &denials, suggestions, span);
                }
            }
            EvalResult::Unmarked => unmarked(span, def_id),
        }

        is_allowed
    }

    pub fn lookup_deprecation(self, id: DefId) -> Option<Deprecation> {
        self.lookup_deprecation_entry(id).map(|depr| depr.attr)
    }
}
