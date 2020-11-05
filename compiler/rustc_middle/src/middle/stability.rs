//! A pass that annotates every item and method with its stability level,
//! propagating default levels lexically from parent to children ast nodes.

pub use self::StabilityLevel::*;

use crate::ty::{self, TyCtxt};
use rustc_ast::CRATE_NODE_ID;
use rustc_attr::{self as attr, ConstStability, Deprecation, Stability};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc_feature::GateIssue;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{CrateNum, DefId, CRATE_DEF_INDEX};
use rustc_hir::{self, HirId};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_session::lint::builtin::{DEPRECATED, DEPRECATED_IN_FUTURE, SOFT_UNSTABLE};
use rustc_session::lint::{BuiltinLintDiagnostics, Lint, LintBuffer};
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

impl StabilityLevel {
    pub fn from_attr_level(level: &attr::StabilityLevel) -> Self {
        if level.is_stable() { Stable } else { Unstable }
    }
}

/// An entry in the `depr_map`.
#[derive(Clone, HashStable)]
pub struct DeprecationEntry {
    /// The metadata of the attribute associated with this entry.
    pub attr: Deprecation,
    /// The `DefId` where the attr was originally attached. `None` for non-local
    /// `DefId`'s.
    origin: Option<HirId>,
}

impl DeprecationEntry {
    pub fn local(attr: Deprecation, id: HirId) -> DeprecationEntry {
        DeprecationEntry { attr, origin: Some(id) }
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
#[derive(HashStable)]
pub struct Index<'tcx> {
    /// This is mostly a cache, except the stabilities of local items
    /// are filled by the annotator.
    pub stab_map: FxHashMap<HirId, &'tcx Stability>,
    pub const_stab_map: FxHashMap<HirId, &'tcx ConstStability>,
    pub depr_map: FxHashMap<HirId, DeprecationEntry>,

    /// Maps for each crate whether it is part of the staged API.
    pub staged_api: FxHashMap<CrateNum, bool>,

    /// Features enabled for this crate.
    pub active_features: FxHashSet<Symbol>,
}

impl<'tcx> Index<'tcx> {
    pub fn local_stability(&self, id: HirId) -> Option<&'tcx Stability> {
        self.stab_map.get(&id).cloned()
    }

    pub fn local_const_stability(&self, id: HirId) -> Option<&'tcx ConstStability> {
        self.const_stab_map.get(&id).cloned()
    }

    pub fn local_deprecation_entry(&self, id: HirId) -> Option<DeprecationEntry> {
        self.depr_map.get(&id).cloned()
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
pub fn deprecation_in_effect(is_since_rustc_version: bool, since: Option<&str>) -> bool {
    let since = if let Some(since) = since {
        if is_since_rustc_version {
            since
        } else {
            // We assume that the deprecation is in effect if it's not a
            // rustc version.
            return true;
        }
    } else {
        // If since attribute is not set, then we're definitely in effect.
        return true;
    };
    fn parse_version(ver: &str) -> Vec<u32> {
        // We ignore non-integer components of the version (e.g., "nightly").
        ver.split(|c| c == '.' || c == '-').flat_map(|s| s.parse()).collect()
    }

    if let Some(rustc) = option_env!("CFG_RELEASE") {
        let since: Vec<u32> = parse_version(&since);
        let rustc: Vec<u32> = parse_version(rustc);
        // We simply treat invalid `since` attributes as relating to a previous
        // Rust version, thus always displaying the warning.
        if since.len() != 3 {
            return true;
        }
        since <= rustc
    } else {
        // By default, a deprecation warning applies to
        // the current version of the compiler.
        true
    }
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

pub fn deprecation_message(depr: &Deprecation, kind: &str, path: &str) -> (String, &'static Lint) {
    let (message, lint) = if deprecation_in_effect(
        depr.is_since_rustc_version,
        depr.since.map(Symbol::as_str).as_deref(),
    ) {
        (format!("use of deprecated {} `{}`", kind, path), DEPRECATED)
    } else {
        (
            format!(
                "use of {} `{}` that will be deprecated in future version {}",
                kind,
                path,
                depr.since.unwrap()
            ),
            DEPRECATED_IN_FUTURE,
        )
    };
    let message = match depr.note {
        Some(reason) => format!("{}: {}", message, reason),
        None => message,
    };
    (message, lint)
}

pub fn early_report_deprecation(
    lint_buffer: &'a mut LintBuffer,
    message: &str,
    suggestion: Option<Symbol>,
    lint: &'static Lint,
    span: Span,
) {
    if span.in_derive_expansion() {
        return;
    }

    let diag = BuiltinLintDiagnostics::DeprecatedMacro(suggestion, span);
    lint_buffer.buffer_lint_with_diagnostic(lint, CRATE_NODE_ID, span, message, diag);
}

fn late_report_deprecation(
    tcx: TyCtxt<'_>,
    message: &str,
    suggestion: Option<Symbol>,
    lint: &'static Lint,
    span: Span,
    hir_id: HirId,
    def_id: DefId,
) {
    if span.in_derive_expansion() {
        return;
    }

    tcx.struct_span_lint_hir(lint, hir_id, span, |lint| {
        let mut diag = lint.build(message);
        if let hir::Node::Expr(_) = tcx.hir().get(hir_id) {
            let kind = tcx.def_kind(def_id).descr(def_id);
            deprecation_suggestion(&mut diag, kind, suggestion, span);
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
    pub fn eval_stability(self, def_id: DefId, id: Option<HirId>, span: Span) -> EvalResult {
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
                if !skip || depr_entry.attr.is_since_rustc_version {
                    let path = &with_no_trimmed_paths(|| self.def_path_str(def_id));
                    let kind = self.def_kind(def_id).descr(def_id);
                    let (message, lint) = deprecation_message(&depr_entry.attr, kind, path);
                    late_report_deprecation(
                        self,
                        &message,
                        depr_entry.attr.suggestion,
                        lint,
                        span,
                        id,
                        def_id,
                    );
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
    pub fn check_stability(self, def_id: DefId, id: Option<HirId>, span: Span) {
        self.check_optional_stability(def_id, id, span, |span, def_id| {
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
        unmarked: impl FnOnce(Span, DefId),
    ) {
        let soft_handler = |lint, span, msg: &_| {
            self.struct_span_lint_hir(lint, id.unwrap_or(hir::CRATE_HIR_ID), span, |lint| {
                lint.build(msg).emit()
            })
        };
        match self.eval_stability(def_id, id, span) {
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
