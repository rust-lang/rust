use std::cmp;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sorted_map::SortedMap;
use rustc_errors::{Diagnostic, DiagnosticBuilder, DiagnosticId, DiagnosticMessage, MultiSpan};
use rustc_hir::{HirId, ItemLocalId};
use rustc_session::lint::{
    builtin::{self, FORBIDDEN_LINT_GROUPS},
    FutureIncompatibilityReason, Level, Lint, LintId,
};
use rustc_session::Session;
use rustc_span::hygiene::MacroKind;
use rustc_span::source_map::{DesugaringKind, ExpnKind};
use rustc_span::{symbol, Span, Symbol, DUMMY_SP};

use crate::ty::TyCtxt;

/// How a lint level was set.
#[derive(Clone, Copy, PartialEq, Eq, HashStable, Debug)]
pub enum LintLevelSource {
    /// Lint is at the default level as declared
    /// in rustc or a plugin.
    Default,

    /// Lint level was set by an attribute.
    Node {
        name: Symbol,
        span: Span,
        /// RFC 2383 reason
        reason: Option<Symbol>,
    },

    /// Lint level was set by a command-line flag.
    /// The provided `Level` is the level specified on the command line.
    /// (The actual level may be lower due to `--cap-lints`.)
    CommandLine(Symbol, Level),
}

impl LintLevelSource {
    pub fn name(&self) -> Symbol {
        match *self {
            LintLevelSource::Default => symbol::kw::Default,
            LintLevelSource::Node { name, .. } => name,
            LintLevelSource::CommandLine(name, _) => name,
        }
    }

    pub fn span(&self) -> Span {
        match *self {
            LintLevelSource::Default => DUMMY_SP,
            LintLevelSource::Node { span, .. } => span,
            LintLevelSource::CommandLine(_, _) => DUMMY_SP,
        }
    }
}

/// A tuple of a lint level and its source.
pub type LevelAndSource = (Level, LintLevelSource);

/// Return type for the `shallow_lint_levels_on` query.
///
/// This map represents the set of allowed lints and allowance levels given
/// by the attributes for *a single HirId*.
#[derive(Default, Debug, HashStable)]
pub struct ShallowLintLevelMap {
    pub specs: SortedMap<ItemLocalId, FxHashMap<LintId, LevelAndSource>>,
}

/// From an initial level and source, verify the effect of special annotations:
/// `warnings` lint level and lint caps.
///
/// The return of this function is suitable for diagnostics.
pub fn reveal_actual_level(
    level: Option<Level>,
    src: &mut LintLevelSource,
    sess: &Session,
    lint: LintId,
    probe_for_lint_level: impl FnOnce(LintId) -> (Option<Level>, LintLevelSource),
) -> Level {
    // If `level` is none then we actually assume the default level for this lint.
    let mut level = level.unwrap_or_else(|| lint.lint.default_level(sess.edition()));

    // If we're about to issue a warning, check at the last minute for any
    // directives against the warnings "lint". If, for example, there's an
    // `allow(warnings)` in scope then we want to respect that instead.
    //
    // We exempt `FORBIDDEN_LINT_GROUPS` from this because it specifically
    // triggers in cases (like #80988) where you have `forbid(warnings)`,
    // and so if we turned that into an error, it'd defeat the purpose of the
    // future compatibility warning.
    if level == Level::Warn && lint != LintId::of(FORBIDDEN_LINT_GROUPS) {
        let (warnings_level, warnings_src) = probe_for_lint_level(LintId::of(builtin::WARNINGS));
        if let Some(configured_warning_level) = warnings_level {
            if configured_warning_level != Level::Warn {
                level = configured_warning_level;
                *src = warnings_src;
            }
        }
    }

    // Ensure that we never exceed the `--cap-lints` argument unless the source is a --force-warn
    level = if let LintLevelSource::CommandLine(_, Level::ForceWarn(_)) = src {
        level
    } else {
        cmp::min(level, sess.opts.lint_cap.unwrap_or(Level::Forbid))
    };

    if let Some(driver_level) = sess.driver_lint_caps.get(&lint) {
        // Ensure that we never exceed driver level.
        level = cmp::min(*driver_level, level);
    }

    level
}

impl ShallowLintLevelMap {
    /// Perform a deep probe in the HIR tree looking for the actual level for the lint.
    /// This lint level is not usable for diagnostics, it needs to be corrected by
    /// `reveal_actual_level` beforehand.
    #[instrument(level = "trace", skip(self, tcx), ret)]
    fn probe_for_lint_level(
        &self,
        tcx: TyCtxt<'_>,
        id: LintId,
        start: HirId,
    ) -> (Option<Level>, LintLevelSource) {
        if let Some(map) = self.specs.get(&start.local_id)
            && let Some(&(level, src)) = map.get(&id)
        {
            return (Some(level), src);
        }

        let mut owner = start.owner;
        let mut specs = &self.specs;

        for parent in tcx.hir().parent_id_iter(start) {
            if parent.owner != owner {
                owner = parent.owner;
                specs = &tcx.shallow_lint_levels_on(owner).specs;
            }
            if let Some(map) = specs.get(&parent.local_id)
                && let Some(&(level, src)) = map.get(&id)
            {
                return (Some(level), src);
            }
        }

        (None, LintLevelSource::Default)
    }

    /// Fetch and return the user-visible lint level for the given lint at the given HirId.
    #[instrument(level = "trace", skip(self, tcx), ret)]
    pub fn lint_level_id_at_node(
        &self,
        tcx: TyCtxt<'_>,
        lint: LintId,
        cur: HirId,
    ) -> (Level, LintLevelSource) {
        let (level, mut src) = self.probe_for_lint_level(tcx, lint, cur);
        let level = reveal_actual_level(level, &mut src, tcx.sess, lint, |lint| {
            self.probe_for_lint_level(tcx, lint, cur)
        });
        (level, src)
    }
}

impl TyCtxt<'_> {
    /// Fetch and return the user-visible lint level for the given lint at the given HirId.
    pub fn lint_level_at_node(self, lint: &'static Lint, id: HirId) -> (Level, LintLevelSource) {
        self.shallow_lint_levels_on(id.owner).lint_level_id_at_node(self, LintId::of(lint), id)
    }

    /// Walks upwards from `id` to find a node which might change lint levels with attributes.
    /// It stops at `bound` and just returns it if reached.
    pub fn maybe_lint_level_root_bounded(self, mut id: HirId, bound: HirId) -> HirId {
        let hir = self.hir();
        loop {
            if id == bound {
                return bound;
            }

            if hir.attrs(id).iter().any(|attr| Level::from_attr(attr).is_some()) {
                return id;
            }
            let next = hir.parent_id(id);
            if next == id {
                bug!("lint traversal reached the root of the crate");
            }
            id = next;
        }
    }
}

/// This struct represents a lint expectation and holds all required information
/// to emit the `unfulfilled_lint_expectations` lint if it is unfulfilled after
/// the `LateLintPass` has completed.
#[derive(Clone, Debug, HashStable)]
pub struct LintExpectation {
    /// The reason for this expectation that can optionally be added as part of
    /// the attribute. It will be displayed as part of the lint message.
    pub reason: Option<Symbol>,
    /// The [`Span`] of the attribute that this expectation originated from.
    pub emission_span: Span,
    /// Lint messages for the `unfulfilled_lint_expectations` lint will be
    /// adjusted to include an additional note. Therefore, we have to track if
    /// the expectation is for the lint.
    pub is_unfulfilled_lint_expectations: bool,
    /// This will hold the name of the tool that this lint belongs to. For
    /// the lint `clippy::some_lint` the tool would be `clippy`, the same
    /// goes for `rustdoc`. This will be `None` for rustc lints
    pub lint_tool: Option<Symbol>,
}

impl LintExpectation {
    pub fn new(
        reason: Option<Symbol>,
        emission_span: Span,
        is_unfulfilled_lint_expectations: bool,
        lint_tool: Option<Symbol>,
    ) -> Self {
        Self { reason, emission_span, is_unfulfilled_lint_expectations, lint_tool }
    }
}

pub fn explain_lint_level_source(
    lint: &'static Lint,
    level: Level,
    src: LintLevelSource,
    err: &mut Diagnostic,
) {
    let name = lint.name_lower();
    match src {
        LintLevelSource::Default => {
            err.note_once(format!("`#[{}({})]` on by default", level.as_str(), name));
        }
        LintLevelSource::CommandLine(lint_flag_val, orig_level) => {
            let flag = orig_level.to_cmd_flag();
            let hyphen_case_lint_name = name.replace('_', "-");
            if lint_flag_val.as_str() == name {
                err.note_once(format!(
                    "requested on the command line with `{} {}`",
                    flag, hyphen_case_lint_name
                ));
            } else {
                let hyphen_case_flag_val = lint_flag_val.as_str().replace('_', "-");
                err.note_once(format!(
                    "`{} {}` implied by `{} {}`",
                    flag, hyphen_case_lint_name, flag, hyphen_case_flag_val
                ));
            }
        }
        LintLevelSource::Node { name: lint_attr_name, span, reason, .. } => {
            if let Some(rationale) = reason {
                err.note(rationale.to_string());
            }
            err.span_note_once(span, "the lint level is defined here");
            if lint_attr_name.as_str() != name {
                let level_str = level.as_str();
                err.note_once(format!(
                    "`#[{}({})]` implied by `#[{}({})]`",
                    level_str, name, level_str, lint_attr_name
                ));
            }
        }
    }
}

/// The innermost function for emitting lints.
///
/// If you are looking to implement a lint, look for higher level functions,
/// for example:
/// - [`TyCtxt::emit_spanned_lint`]
/// - [`TyCtxt::struct_span_lint_hir`]
/// - [`TyCtxt::emit_lint`]
/// - [`TyCtxt::struct_lint_node`]
/// - `LintContext::lookup`
///
/// ## `decorate` signature
///
/// The return value of `decorate` is ignored by this function. So what is the
/// point of returning `&'b mut DiagnosticBuilder<'a, ()>`?
///
/// There are 2 reasons for this signature.
///
/// First of all, it prevents accidental use of `.emit()` -- it's clear that the
/// builder will be later used and shouldn't be emitted right away (this is
/// especially important because the old API expected you to call `.emit()` in
/// the closure).
///
/// Second of all, it makes the most common case of adding just a single label
/// /suggestion much nicer, since [`DiagnosticBuilder`] methods return
/// `&mut DiagnosticBuilder`, you can just chain methods, without needed
/// awkward `{ ...; }`:
/// ```ignore pseudo-code
/// struct_lint_level(
///     ...,
///     |lint| lint.span_label(sp, "lbl")
///     //          ^^^^^^^^^^^^^^^^^^^^^ returns `&mut DiagnosticBuilder` by default
/// )
/// ```
pub fn struct_lint_level(
    sess: &Session,
    lint: &'static Lint,
    level: Level,
    src: LintLevelSource,
    span: Option<MultiSpan>,
    msg: impl Into<DiagnosticMessage>,
    decorate: impl for<'a, 'b> FnOnce(
        &'b mut DiagnosticBuilder<'a, ()>,
    ) -> &'b mut DiagnosticBuilder<'a, ()>,
) {
    // Avoid codegen bloat from monomorphization by immediately doing dyn dispatch of `decorate` to
    // the "real" work.
    fn struct_lint_level_impl(
        sess: &Session,
        lint: &'static Lint,
        level: Level,
        src: LintLevelSource,
        span: Option<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
        decorate: Box<
            dyn '_
                + for<'a, 'b> FnOnce(
                    &'b mut DiagnosticBuilder<'a, ()>,
                ) -> &'b mut DiagnosticBuilder<'a, ()>,
        >,
    ) {
        // Check for future incompatibility lints and issue a stronger warning.
        let future_incompatible = lint.future_incompatible;

        let has_future_breakage = future_incompatible.map_or(
            // Default allow lints trigger too often for testing.
            sess.opts.unstable_opts.future_incompat_test && lint.default_level != Level::Allow,
            |incompat| {
                matches!(incompat.reason, FutureIncompatibilityReason::FutureReleaseErrorReportNow)
            },
        );

        let mut err = match (level, span) {
            (Level::Allow, span) => {
                if has_future_breakage {
                    if let Some(span) = span {
                        sess.struct_span_allow(span, "")
                    } else {
                        sess.struct_allow("")
                    }
                } else {
                    return;
                }
            }
            (Level::Expect(expect_id), _) => {
                // This case is special as we actually allow the lint itself in this context, but
                // we can't return early like in the case for `Level::Allow` because we still
                // need the lint diagnostic to be emitted to `rustc_error::HandlerInner`.
                //
                // We can also not mark the lint expectation as fulfilled here right away, as it
                // can still be cancelled in the decorate function. All of this means that we simply
                // create a `DiagnosticBuilder` and continue as we would for warnings.
                sess.struct_expect("", expect_id)
            }
            (Level::ForceWarn(Some(expect_id)), Some(span)) => {
                sess.struct_span_warn_with_expectation(span, "", expect_id)
            }
            (Level::ForceWarn(Some(expect_id)), None) => {
                sess.struct_warn_with_expectation("", expect_id)
            }
            (Level::Warn | Level::ForceWarn(None), Some(span)) => sess.struct_span_warn(span, ""),
            (Level::Warn | Level::ForceWarn(None), None) => sess.struct_warn(""),
            (Level::Deny | Level::Forbid, Some(span)) => {
                let mut builder = sess.diagnostic().struct_err_lint("");
                builder.set_span(span);
                builder
            }
            (Level::Deny | Level::Forbid, None) => sess.diagnostic().struct_err_lint(""),
        };

        err.set_is_lint();

        // If this code originates in a foreign macro, aka something that this crate
        // did not itself author, then it's likely that there's nothing this crate
        // can do about it. We probably want to skip the lint entirely.
        if err.span.primary_spans().iter().any(|s| in_external_macro(sess, *s)) {
            // Any suggestions made here are likely to be incorrect, so anything we
            // emit shouldn't be automatically fixed by rustfix.
            err.disable_suggestions();

            // If this is a future incompatible that is not an edition fixing lint
            // it'll become a hard error, so we have to emit *something*. Also,
            // if this lint occurs in the expansion of a macro from an external crate,
            // allow individual lints to opt-out from being reported.
            let not_future_incompatible =
                future_incompatible.map(|f| f.reason.edition().is_some()).unwrap_or(true);
            if not_future_incompatible && !lint.report_in_external_macro {
                err.cancel();
                // Don't continue further, since we don't want to have
                // `diag_span_note_once` called for a diagnostic that isn't emitted.
                return;
            }
        }

        // Delay evaluating and setting the primary message until after we've
        // suppressed the lint due to macros.
        err.set_primary_message(msg);

        // Lint diagnostics that are covered by the expect level will not be emitted outside
        // the compiler. It is therefore not necessary to add any information for the user.
        // This will therefore directly call the decorate function which will in turn emit
        // the `Diagnostic`.
        if let Level::Expect(_) = level {
            let name = lint.name_lower();
            err.code(DiagnosticId::Lint { name, has_future_breakage, is_force_warn: false });

            decorate(&mut err);
            err.emit();
            return;
        }

        let name = lint.name_lower();
        let is_force_warn = matches!(level, Level::ForceWarn(_));
        err.code(DiagnosticId::Lint { name, has_future_breakage, is_force_warn });

        if let Some(future_incompatible) = future_incompatible {
            let explanation = match future_incompatible.reason {
                FutureIncompatibilityReason::FutureReleaseError
                | FutureIncompatibilityReason::FutureReleaseErrorReportNow => {
                    "this was previously accepted by the compiler but is being phased out; \
                         it will become a hard error in a future release!"
                        .to_owned()
                }
                FutureIncompatibilityReason::FutureReleaseSemanticsChange => {
                    "this will change its meaning in a future release!".to_owned()
                }
                FutureIncompatibilityReason::EditionError(edition) => {
                    let current_edition = sess.edition();
                    format!(
                        "this is accepted in the current edition (Rust {}) but is a hard error in Rust {}!",
                        current_edition, edition
                    )
                }
                FutureIncompatibilityReason::EditionSemanticsChange(edition) => {
                    format!("this changes meaning in Rust {}", edition)
                }
                FutureIncompatibilityReason::Custom(reason) => reason.to_owned(),
            };

            if future_incompatible.explain_reason {
                err.warn(explanation);
            }
            if !future_incompatible.reference.is_empty() {
                let citation =
                    format!("for more information, see {}", future_incompatible.reference);
                err.note(citation);
            }
        }

        // Finally, run `decorate`.
        decorate(&mut err);
        explain_lint_level_source(lint, level, src, &mut *err);
        err.emit()
    }
    struct_lint_level_impl(sess, lint, level, src, span, msg, Box::new(decorate))
}

/// Returns whether `span` originates in a foreign crate's external macro.
///
/// This is used to test whether a lint should not even begin to figure out whether it should
/// be reported on the current node.
pub fn in_external_macro(sess: &Session, span: Span) -> bool {
    let expn_data = span.ctxt().outer_expn_data();
    match expn_data.kind {
        ExpnKind::Root
        | ExpnKind::Desugaring(
            DesugaringKind::ForLoop | DesugaringKind::WhileLoop | DesugaringKind::OpaqueTy,
        ) => false,
        ExpnKind::AstPass(_) | ExpnKind::Desugaring(_) => true, // well, it's "external"
        ExpnKind::Macro(MacroKind::Bang, _) => {
            // Dummy span for the `def_site` means it's an external macro.
            expn_data.def_site.is_dummy() || sess.source_map().is_imported(expn_data.def_site)
        }
        ExpnKind::Macro { .. } => true, // definitely a plugin
    }
}
