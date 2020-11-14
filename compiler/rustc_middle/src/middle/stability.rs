//! A pass that annotates every item and method with its stability level,
//! propagating default levels lexically from parent to children ast nodes.

pub use self::StabilityLevel::*;

use crate::ty::{self, TyCtxt};
use rustc_attr::{self as attr, Deprecation, Stability};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, CRATE_DEF_INDEX};
use rustc_hir::{self, HirId};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_session::lint::Lint;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::Span;

use std::num::NonZeroU32;

use rustc_crate::stability::{
    deprecation_message, deprecation_suggestion, report_unstable, StabilityLevel,
};

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
