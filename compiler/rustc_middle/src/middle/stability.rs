//! A pass that annotates every item and method with its stability level,
//! propagating default levels lexically from parent to children ast nodes.

pub use self::StabilityLevel::*;

use crate::ty::{self, TyCtxt};
use rustc_ast::NodeId;
use rustc_attr::{self as attr, ConstStability, Deprecation, Stability};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc_feature::GateIssue;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{CrateNum, DefId, LocalDefId, CRATE_DEF_INDEX};
use rustc_hir::{self, HirId};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_session::lint::builtin::{DEPRECATED, DEPRECATED_IN_FUTURE, SOFT_UNSTABLE};
use rustc_session::lint::{BuiltinLintDiagnostics, Level, Lint, LintBuffer};
use rustc_session::parse::feature_err_issue;
use rustc_session::{DiagnosticMessageId, Session};
use rustc_span::symbol::{sym, Symbol};
use rustc_span::{MultiSpan, Span};
use std::num::NonZeroU32;

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum StabilityLevel {
    Unstable,
    Stable,
}

/// An entry in the `depr_map`.
#[derive(Clone, HashStable, Debug)]
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
    pub stab_map: FxHashMap<LocalDefId, &'tcx Stability>,
    pub const_stab_map: FxHashMap<LocalDefId, &'tcx ConstStability>,
    pub depr_map: FxHashMap<LocalDefId, DeprecationEntry>,

    /// Maps for each crate whether it is part of the staged API.
    pub staged_api: FxHashMap<CrateNum, bool>,

    /// Features enabled for this crate.
    pub active_features: FxHashSet<Symbol>,
}

impl<'tcx> Index<'tcx> {
    pub fn local_stability(&self, def_id: LocalDefId) -> Option<&'tcx Stability> {
        self.stab_map.get(&def_id).copied()
    }

    pub fn local_const_stability(&self, def_id: LocalDefId) -> Option<&'tcx ConstStability> {
        self.const_stab_map.get(&def_id).copied()
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
    is_soft: bool,
    span: Span,
    soft_handler: impl FnOnce(&'static Lint, Span, &str),
) {
    let msg = match reason {
        Some(r) => format!("use of unstable library feature '{}': {}", feature, r),
        None => format!("use of unstable library feature '{}'", &feature),
    };

    let msp: MultiSpan = span.into();
    let sm = &sess.parse_sess.source_map();
    let span_key = msp.primary_span().and_then(|sp: Span| {
        if !sp.is_dummy() {
            let file = sm.lookup_char_pos(sp.lo()).file;
            if file.is_imported() { None } else { Some(span) }
        } else {
            None
        }
    });

    let error_id = (DiagnosticMessageId::StabilityId(issue), span_key, msg.clone());
    let fresh = sess.one_time_diagnostics.borrow_mut().insert(error_id);
    if fresh {
        if is_soft {
            soft_handler(SOFT_UNSTABLE, span, &msg)
        } else {
            feature_err_issue(&sess.parse_sess, feature, span, GateIssue::Library(issue), &msg)
                .emit();
        }
    }
}

/// Checks whether an item marked with `deprecated(since="X")` is currently
/// deprecated (i.e., whether X is not greater than the current rustc version).
pub fn deprecation_in_effect(depr: &Deprecation) -> bool {
    let is_since_rustc_version = depr.is_since_rustc_version;
    let since = depr.since.map(Symbol::as_str);
    let since = since.as_deref();

    fn parse_version(ver: &str) -> Vec<u32> {
        // We ignore non-integer components of the version (e.g., "nightly").
        ver.split(|c| c == '.' || c == '-').flat_map(|s| s.parse()).collect()
    }

    if !is_since_rustc_version {
        // The `since` field doesn't have semantic purpose in the stable `deprecated`
        // attribute, only in `rustc_deprecated`.
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
    diag: &mut DiagnosticBuilder<'_>,
    kind: &str,
    suggestion: Option<Symbol>,
    span: Span,
) {
    if let Some(suggestion) = suggestion {
        diag.span_suggestion(
            span,
            &format!("replace the use of the deprecated {}", kind),
            suggestion.to_string(),
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
        let since = since.map(Symbol::as_str);

        if since.as_deref() == Some("TBD") {
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
    lint_buffer: &'a mut LintBuffer,
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
    tcx.struct_span_lint_hir(lint, hir_id, method_span, |lint| {
        let mut diag = lint.build(message);
        if let hir::Node::Expr(_) = tcx.hir().get(hir_id) {
            let kind = tcx.def_kind(def_id).descr(def_id);
            deprecation_suggestion(&mut diag, kind, suggestion, method_span);
        }
        diag.emit()
    });
}

/// Result of `TyCtxt::eval_stability`.
pub enum EvalResult {
    /// We can use the item because it is stable or we provided the
    /// corresponding feature gate.
    Allow,
    /// We cannot use the item because it is unstable and we did not provide the
    /// corresponding feature gate.
    Deny { feature: Symbol, reason: Option<Symbol>, issue: Option<NonZeroU32>, is_soft: bool },
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
        ty::Visibility::Restricted(..) | ty::Visibility::Invisible => true,
    }
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
        // Deprecated attributes apply in-crate and cross-crate.
        if let Some(id) = id {
            if let Some(depr_entry) = self.lookup_deprecation_entry(def_id) {
                let parent_def_id = self.hir().local_def_id(self.hir().get_parent_item(id));
                let skip = self
                    .lookup_deprecation_entry(parent_def_id.to_def_id())
                    .map_or(false, |parent_depr| parent_depr.same_origin(&depr_entry));

                // #[deprecated] doesn't emit a notice if we're not on the
                // topmost deprecation. For example, if a struct is deprecated,
                // the use of a field won't be linted.
                //
                // #[rustc_deprecated] however wants to emit down the whole
                // hierarchy.
                let depr_attr = &depr_entry.attr;
                if !skip || depr_attr.is_since_rustc_version {
                    // Calculating message for lint involves calling `self.def_path_str`.
                    // Which by default to calculate visible path will invoke expensive `visible_parent_map` query.
                    // So we skip message calculation altogether, if lint is allowed.
                    let is_in_effect = deprecation_in_effect(depr_attr);
                    let lint = deprecation_lint(is_in_effect);
                    if self.lint_level_at_node(lint, id).0 != Level::Allow {
                        let def_path = &with_no_trimmed_paths(|| self.def_path_str(def_id));
                        let def_kind = self.def_kind(def_id).descr(def_id);

                        late_report_deprecation(
                            self,
                            &deprecation_message(
                                is_in_effect,
                                depr_attr.since,
                                depr_attr.note,
                                def_kind,
                                def_path,
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

        let is_staged_api =
            self.lookup_stability(DefId { index: CRATE_DEF_INDEX, ..def_id }).is_some();
        if !is_staged_api {
            return EvalResult::Allow;
        }

        let stability = self.lookup_stability(def_id);
        debug!(
            "stability: \
                inspecting def_id={:?} span={:?} of stability={:?}",
            def_id, span, stability
        );

        // Only the cross-crate scenario matters when checking unstable APIs
        let cross_crate = !def_id.is_local();
        if !cross_crate {
            return EvalResult::Allow;
        }

        // Issue #38412: private items lack stability markers.
        if skip_stability_check_due_to_privacy(self, def_id) {
            return EvalResult::Allow;
        }

        match stability {
            Some(&Stability {
                level: attr::Unstable { reason, issue, is_soft }, feature, ..
            }) => {
                if span.allows_unstable(feature) {
                    debug!("stability: skipping span={:?} since it is internal", span);
                    return EvalResult::Allow;
                }
                if self.stability().active_features.contains(&feature) {
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
                    if self.sess.opts.debugging_opts.force_unstable_if_unmarked {
                        return EvalResult::Allow;
                    }
                }

                EvalResult::Deny { feature, reason, issue, is_soft }
            }
            Some(_) => {
                // Stable APIs are always ok to call and deprecated APIs are
                // handled by the lint emitting logic above.
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
    pub fn check_stability(
        self,
        def_id: DefId,
        id: Option<HirId>,
        span: Span,
        method_span: Option<Span>,
    ) {
        self.check_optional_stability(def_id, id, span, method_span, |span, def_id| {
            // The API could be uncallable for other reasons, for example when a private module
            // was referenced.
            self.sess.delay_span_bug(span, &format!("encountered unmarked API: {:?}", def_id));
        })
    }

    /// Like `check_stability`, except that we permit items to have custom behaviour for
    /// missing stability attributes (not necessarily just emit a `bug!`). This is necessary
    /// for default generic parameters, which only have stability attributes if they were
    /// added after the type on which they're defined.
    pub fn check_optional_stability(
        self,
        def_id: DefId,
        id: Option<HirId>,
        span: Span,
        method_span: Option<Span>,
        unmarked: impl FnOnce(Span, DefId),
    ) {
        let soft_handler = |lint, span, msg: &_| {
            self.struct_span_lint_hir(lint, id.unwrap_or(hir::CRATE_HIR_ID), span, |lint| {
                lint.build(msg).emit()
            })
        };
        match self.eval_stability(def_id, id, span, method_span) {
            EvalResult::Allow => {}
            EvalResult::Deny { feature, reason, issue, is_soft } => {
                report_unstable(self.sess, feature, reason, issue, is_soft, span, soft_handler)
            }
            EvalResult::Unmarked => unmarked(span, def_id),
        }
    }

    pub fn lookup_deprecation(self, id: DefId) -> Option<Deprecation> {
        self.lookup_deprecation_entry(id).map(|depr| depr.attr)
    }
}
