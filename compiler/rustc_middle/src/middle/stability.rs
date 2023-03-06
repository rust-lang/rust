//! A pass that annotates every item and method with its stability level,
//! propagating default levels lexically from parent to children ast nodes.

pub use self::StabilityLevel::*;

use crate::ty::{self, TyCtxt};
use rustc_ast::NodeId;
use rustc_attr::{self as attr, ConstStability, DefaultBodyStability, Deprecation, Stability};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{Applicability, Diagnostic};
use rustc_feature::GateIssue;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{self as hir, HirId};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_session::lint::builtin::{DEPRECATED, DEPRECATED_IN_FUTURE, SOFT_UNSTABLE};
use rustc_session::lint::{BuiltinLintDiagnostics, Level, Lint, LintBuffer};
use rustc_session::parse::feature_err_issue;
use rustc_session::Session;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::Span;
use std::num::NonZeroU32;

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
pub struct Index {
    /// This is mostly a cache, except the stabilities of local items
    /// are filled by the annotator.
    pub stab_map: FxHashMap<LocalDefId, Stability>,
    pub const_stab_map: FxHashMap<LocalDefId, ConstStability>,
    pub default_body_stab_map: FxHashMap<LocalDefId, DefaultBodyStability>,
    pub depr_map: FxHashMap<LocalDefId, DeprecationEntry>,
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
    pub implications: FxHashMap<Symbol, Symbol>,
}

impl Index {
    pub fn local_stability(&self, def_id: LocalDefId) -> Option<Stability> {
        self.stab_map.get(&def_id).copied()
    }

    pub fn local_const_stability(&self, def_id: LocalDefId) -> Option<ConstStability> {
        self.const_stab_map.get(&def_id).copied()
    }

    pub fn local_default_body_stability(&self, def_id: LocalDefId) -> Option<DefaultBodyStability> {
        self.default_body_stab_map.get(&def_id).copied()
    }

    pub fn local_deprecation_entry(&self, def_id: LocalDefId) -> Option<DeprecationEntry> {
        self.depr_map.get(&def_id).cloned()
    }
}

pub fn report_unstable(
    sess: &Session,
    feature: Symbol,
    reason: Option<Symbol>,
    issue: Option<NonZeroU32>,
    suggestion: Option<(Span, String, String, Applicability)>,
    is_soft: bool,
    span: Span,
    soft_handler: impl FnOnce(&'static Lint, Span, &str),
) {
    let msg = match reason {
        Some(r) => format!("use of unstable library feature '{}': {}", feature, r),
        None => format!("use of unstable library feature '{}'", &feature),
    };

    if is_soft {
        soft_handler(SOFT_UNSTABLE, span, &msg)
    } else {
        let mut err =
            feature_err_issue(&sess.parse_sess, feature, span, GateIssue::Library(issue), &msg);
        if let Some((inner_types, ref msg, sugg, applicability)) = suggestion {
            err.span_suggestion(inner_types, msg, sugg, applicability);
        }
        err.emit();
    }
}

/// Checks whether an item marked with `deprecated(since="X")` is currently
/// deprecated (i.e., whether X is not greater than the current rustc version).
pub fn deprecation_in_effect(depr: &Deprecation) -> bool {
    let is_since_rustc_version = depr.is_since_rustc_version;
    let since = depr.since.as_ref().map(Symbol::as_str);

    fn parse_version(ver: &str) -> Vec<u32> {
        // We ignore non-integer components of the version (e.g., "nightly").
        ver.split(|c| c == '.' || c == '-').flat_map(|s| s.parse()).collect()
    }

    if !is_since_rustc_version {
        // The `since` field doesn't have semantic purpose without `#![staged_api]`.
        return true;
    }

    if let Some(since) = since {
        if since == "TBD" {
            return false;
        }

        if let Some(rustc) = option_env!("CFG_RELEASE") {
            let since: Vec<u32> = parse_version(&since);
            let rustc: Vec<u32> = parse_version(rustc);
            // We simply treat invalid `since` attributes as relating to a previous
            // Rust version, thus always displaying the warning.
            if since.len() != 3 {
                return true;
            }
            return since <= rustc;
        }
    };

    // Assume deprecation is in effect if "since" field is missing
    // or if we can't determine the current Rust version.
    true
}

pub fn deprecation_suggestion(
    diag: &mut Diagnostic,
    kind: &str,
    suggestion: Option<Symbol>,
    span: Span,
) {
    if let Some(suggestion) = suggestion {
        diag.span_suggestion_verbose(
            span,
            &format!("replace the use of the deprecated {}", kind),
            suggestion,
            Applicability::MachineApplicable,
        );
    }
}

fn deprecation_lint(is_in_effect: bool) -> &'static Lint {
    if is_in_effect { DEPRECATED } else { DEPRECATED_IN_FUTURE }
}

fn deprecation_message(
    is_in_effect: bool,
    since: Option<Symbol>,
    note: Option<Symbol>,
    kind: &str,
    path: &str,
) -> String {
    let message = if is_in_effect {
        format!("use of deprecated {} `{}`", kind, path)
    } else {
        let since = since.as_ref().map(Symbol::as_str);

        if since == Some("TBD") {
            format!("use of {} `{}` that will be deprecated in a future Rust version", kind, path)
        } else {
            format!(
                "use of {} `{}` that will be deprecated in future version {}",
                kind,
                path,
                since.unwrap()
            )
        }
    };

    match note {
        Some(reason) => format!("{}: {}", message, reason),
        None => message,
    }
}

pub fn deprecation_message_and_lint(
    depr: &Deprecation,
    kind: &str,
    path: &str,
) -> (String, &'static Lint) {
    let is_in_effect = deprecation_in_effect(depr);
    (
        deprecation_message(is_in_effect, depr.since, depr.note, kind, path),
        deprecation_lint(is_in_effect),
    )
}

pub fn early_report_deprecation(
    lint_buffer: &mut LintBuffer,
    message: &str,
    suggestion: Option<Symbol>,
    lint: &'static Lint,
    span: Span,
    node_id: NodeId,
) {
    if span.in_derive_expansion() {
        return;
    }

    let diag = BuiltinLintDiagnostics::DeprecatedMacro(suggestion, span);
    lint_buffer.buffer_lint_with_diagnostic(lint, node_id, span, message, diag);
}

fn late_report_deprecation(
    tcx: TyCtxt<'_>,
    message: &str,
    suggestion: Option<Symbol>,
    lint: &'static Lint,
    span: Span,
    method_span: Option<Span>,
    hir_id: HirId,
    def_id: DefId,
) {
    if span.in_derive_expansion() {
        return;
    }
    let method_span = method_span.unwrap_or(span);
    tcx.struct_span_lint_hir(lint, hir_id, method_span, message, |diag| {
        if let hir::Node::Expr(_) = tcx.hir().get(hir_id) {
            let kind = tcx.def_descr(def_id);
            deprecation_suggestion(diag, kind, suggestion, method_span);
        }
        diag
    });
}

/// Result of `TyCtxt::eval_stability`.
pub enum EvalResult {
    /// We can use the item because it is stable or we provided the
    /// corresponding feature gate.
    Allow,
    /// We cannot use the item because it is unstable and we did not provide the
    /// corresponding feature gate.
    Deny {
        feature: Symbol,
        reason: Option<Symbol>,
        issue: Option<NonZeroU32>,
        suggestion: Option<(Span, String, String, Applicability)>,
        is_soft: bool,
    },
    /// The item does not have the `#[stable]` or `#[unstable]` marker assigned.
    Unmarked,
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
) -> Option<(Span, String, String, Applicability)> {
    if feature == sym::allocator_api {
        if let Some(trait_) = tcx.opt_parent(def_id) {
            if tcx.is_diagnostic_item(sym::Vec, trait_) {
                let sm = tcx.sess.parse_sess.source_map();
                let inner_types = sm.span_extend_to_prev_char(span, '<', true);
                if let Ok(snippet) = sm.span_to_snippet(inner_types) {
                    return Some((
                        inner_types,
                        "consider wrapping the inner types in tuple".to_string(),
                        format!("({})", snippet),
                        Applicability::MaybeIncorrect,
                    ));
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
    /// `#![feature]` has been provided. Returns `EvalResult::Deny` which describes the offending
    /// unstable feature otherwise.
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
                    .map_or(false, |parent_depr| parent_depr.same_origin(&depr_entry));

                // #[deprecated] doesn't emit a notice if we're not on the
                // topmost deprecation. For example, if a struct is deprecated,
                // the use of a field won't be linted.
                //
                // With #![staged_api], we want to emit down the whole
                // hierarchy.
                let depr_attr = &depr_entry.attr;
                if !skip || depr_attr.is_since_rustc_version {
                    // Calculating message for lint involves calling `self.def_path_str`.
                    // Which by default to calculate visible path will invoke expensive `visible_parent_map` query.
                    // So we skip message calculation altogether, if lint is allowed.
                    let is_in_effect = deprecation_in_effect(depr_attr);
                    let lint = deprecation_lint(is_in_effect);
                    if self.lint_level_at_node(lint, id).0 != Level::Allow {
                        let def_path = with_no_trimmed_paths!(self.def_path_str(def_id));
                        let def_kind = self.def_descr(def_id);

                        late_report_deprecation(
                            self,
                            &deprecation_message(
                                is_in_effect,
                                depr_attr.since,
                                depr_attr.note,
                                def_kind,
                                &def_path,
                            ),
                            depr_attr.suggestion,
                            lint,
                            span,
                            method_span,
                            id,
                            def_id,
                        );
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
            Some(Stability {
                level: attr::Unstable { reason, issue, is_soft, implied_by },
                feature,
                ..
            }) => {
                if span.allows_unstable(feature) {
                    debug!("stability: skipping span={:?} since it is internal", span);
                    return EvalResult::Allow;
                }
                if self.features().active(feature) {
                    return EvalResult::Allow;
                }

                // If this item was previously part of a now-stabilized feature which is still
                // active (i.e. the user hasn't removed the attribute for the stabilized feature
                // yet) then allow use of this item.
                if let Some(implied_by) = implied_by && self.features().active(implied_by) {
                    return EvalResult::Allow;
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
                if feature == sym::rustc_private && issue == NonZeroU32::new(27812) {
                    if self.sess.opts.unstable_opts.force_unstable_if_unmarked {
                        return EvalResult::Allow;
                    }
                }

                if matches!(allow_unstable, AllowUnstable::Yes) {
                    return EvalResult::Allow;
                }

                let suggestion = suggestion_for_allocator_api(self, def_id, span, feature);
                EvalResult::Deny {
                    feature,
                    reason: reason.to_opt_reason(),
                    issue,
                    suggestion,
                    is_soft,
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
    /// `#![feature]` has been provided. Returns `EvalResult::Deny` which describes the offending
    /// unstable feature otherwise.
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
            Some(DefaultBodyStability {
                level: attr::Unstable { reason, issue, is_soft, .. },
                feature,
            }) => {
                if span.allows_unstable(feature) {
                    debug!("body stability: skipping span={:?} since it is internal", span);
                    return EvalResult::Allow;
                }
                if self.features().active(feature) {
                    return EvalResult::Allow;
                }

                EvalResult::Deny {
                    feature,
                    reason: reason.to_opt_reason(),
                    issue,
                    suggestion: None,
                    is_soft,
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
                self.sess.delay_span_bug(span, &format!("encountered unmarked API: {:?}", def_id));
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
        let soft_handler = |lint, span, msg: &_| {
            self.struct_span_lint_hir(lint, id.unwrap_or(hir::CRATE_HIR_ID), span, msg, |lint| lint)
        };
        let eval_result =
            self.eval_stability_allow_unstable(def_id, id, span, method_span, allow_unstable);
        let is_allowed = matches!(eval_result, EvalResult::Allow);
        match eval_result {
            EvalResult::Allow => {}
            EvalResult::Deny { feature, reason, issue, suggestion, is_soft } => report_unstable(
                self.sess,
                feature,
                reason,
                issue,
                suggestion,
                is_soft,
                span,
                soft_handler,
            ),
            EvalResult::Unmarked => unmarked(span, def_id),
        }

        is_allowed
    }

    pub fn lookup_deprecation(self, id: DefId) -> Option<Deprecation> {
        self.lookup_deprecation_entry(id).map(|depr| depr.attr)
    }
}
