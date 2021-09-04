use std::cmp;

use crate::ich::StableHashingContext;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_errors::{DiagnosticBuilder, DiagnosticId};
use rustc_hir::HirId;
use rustc_index::vec::IndexVec;
use rustc_session::lint::{
    builtin::{self, FORBIDDEN_LINT_GROUPS},
    FutureIncompatibilityReason, Level, Lint, LintId,
};
use rustc_session::{DiagnosticMessageId, Session};
use rustc_span::hygiene::MacroKind;
use rustc_span::source_map::{DesugaringKind, ExpnKind, MultiSpan};
use rustc_span::{symbol, Span, Symbol, DUMMY_SP};

/// How a lint level was set.
#[derive(Clone, Copy, PartialEq, Eq, HashStable, Debug)]
pub enum LintLevelSource {
    /// Lint is at the default level as declared
    /// in rustc or a plugin.
    Default,

    /// Lint level was set by an attribute.
    Node(Symbol, Span, Option<Symbol> /* RFC 2383 reason */),

    /// Lint level was set by a command-line flag.
    /// The provided `Level` is the level specified on the command line.
    /// (The actual level may be lower due to `--cap-lints`.)
    CommandLine(Symbol, Level),
}

impl LintLevelSource {
    pub fn name(&self) -> Symbol {
        match *self {
            LintLevelSource::Default => symbol::kw::Default,
            LintLevelSource::Node(name, _, _) => name,
            LintLevelSource::CommandLine(name, _) => name,
        }
    }

    pub fn span(&self) -> Span {
        match *self {
            LintLevelSource::Default => DUMMY_SP,
            LintLevelSource::Node(_, span, _) => span,
            LintLevelSource::CommandLine(_, _) => DUMMY_SP,
        }
    }
}

/// A tuple of a lint level and its source.
pub type LevelAndSource = (Level, LintLevelSource);

#[derive(Debug, HashStable)]
pub struct LintLevelSets {
    pub list: IndexVec<LintStackIndex, LintSet>,
    pub lint_cap: Level,
}

rustc_index::newtype_index! {
    #[derive(HashStable)]
    pub struct LintStackIndex {
        const COMMAND_LINE = 0,
    }
}

#[derive(Debug, HashStable)]
pub struct LintSet {
    // -A,-W,-D flags, a `Symbol` for the flag itself and `Level` for which
    // flag.
    pub specs: FxHashMap<LintId, LevelAndSource>,

    pub parent: LintStackIndex,
}

impl LintLevelSets {
    pub fn new() -> Self {
        LintLevelSets { list: IndexVec::new(), lint_cap: Level::Forbid }
    }

    pub fn get_lint_level(
        &self,
        lint: &'static Lint,
        idx: LintStackIndex,
        aux: Option<&FxHashMap<LintId, LevelAndSource>>,
        sess: &Session,
    ) -> LevelAndSource {
        let (level, mut src) = self.get_lint_id_level(LintId::of(lint), idx, aux);

        // If `level` is none then we actually assume the default level for this
        // lint.
        let mut level = level.unwrap_or_else(|| lint.default_level(sess.edition()));

        // If we're about to issue a warning, check at the last minute for any
        // directives against the warnings "lint". If, for example, there's an
        // `allow(warnings)` in scope then we want to respect that instead.
        //
        // We exempt `FORBIDDEN_LINT_GROUPS` from this because it specifically
        // triggers in cases (like #80988) where you have `forbid(warnings)`,
        // and so if we turned that into an error, it'd defeat the purpose of the
        // future compatibility warning.
        if level == Level::Warn && LintId::of(lint) != LintId::of(FORBIDDEN_LINT_GROUPS) {
            let (warnings_level, warnings_src) =
                self.get_lint_id_level(LintId::of(builtin::WARNINGS), idx, aux);
            if let Some(configured_warning_level) = warnings_level {
                if configured_warning_level != Level::Warn {
                    level = configured_warning_level;
                    src = warnings_src;
                }
            }
        }

        // Ensure that we never exceed the `--cap-lints` argument
        // unless the source is a --force-warn
        level = if let LintLevelSource::CommandLine(_, Level::ForceWarn) = src {
            level
        } else {
            cmp::min(level, self.lint_cap)
        };

        if let Some(driver_level) = sess.driver_lint_caps.get(&LintId::of(lint)) {
            // Ensure that we never exceed driver level.
            level = cmp::min(*driver_level, level);
        }

        (level, src)
    }

    pub fn get_lint_id_level(
        &self,
        id: LintId,
        mut idx: LintStackIndex,
        aux: Option<&FxHashMap<LintId, LevelAndSource>>,
    ) -> (Option<Level>, LintLevelSource) {
        if let Some(specs) = aux {
            if let Some(&(level, src)) = specs.get(&id) {
                return (Some(level), src);
            }
        }
        loop {
            let LintSet { ref specs, parent } = self.list[idx];
            if let Some(&(level, src)) = specs.get(&id) {
                return (Some(level), src);
            }
            if idx == COMMAND_LINE {
                return (None, LintLevelSource::Default);
            }
            idx = parent;
        }
    }
}

#[derive(Debug)]
pub struct LintLevelMap {
    pub sets: LintLevelSets,
    pub id_to_set: FxHashMap<HirId, LintStackIndex>,
}

impl LintLevelMap {
    /// If the `id` was previously registered with `register_id` when building
    /// this `LintLevelMap` this returns the corresponding lint level and source
    /// of the lint level for the lint provided.
    ///
    /// If the `id` was not previously registered, returns `None`. If `None` is
    /// returned then the parent of `id` should be acquired and this function
    /// should be called again.
    pub fn level_and_source(
        &self,
        lint: &'static Lint,
        id: HirId,
        session: &Session,
    ) -> Option<LevelAndSource> {
        self.id_to_set.get(&id).map(|idx| self.sets.get_lint_level(lint, *idx, None, session))
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for LintLevelMap {
    #[inline]
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        let LintLevelMap { ref sets, ref id_to_set } = *self;

        id_to_set.hash_stable(hcx, hasher);

        hcx.while_hashing_spans(true, |hcx| sets.hash_stable(hcx, hasher))
    }
}

pub struct LintDiagnosticBuilder<'a>(DiagnosticBuilder<'a>);

impl<'a> LintDiagnosticBuilder<'a> {
    /// Return the inner DiagnosticBuilder, first setting the primary message to `msg`.
    pub fn build(mut self, msg: &str) -> DiagnosticBuilder<'a> {
        self.0.set_primary_message(msg);
        self.0.set_is_lint();
        self.0
    }

    /// Create a LintDiagnosticBuilder from some existing DiagnosticBuilder.
    pub fn new(err: DiagnosticBuilder<'a>) -> LintDiagnosticBuilder<'a> {
        LintDiagnosticBuilder(err)
    }
}

pub fn struct_lint_level<'s, 'd>(
    sess: &'s Session,
    lint: &'static Lint,
    level: Level,
    src: LintLevelSource,
    span: Option<MultiSpan>,
    decorate: impl for<'a> FnOnce(LintDiagnosticBuilder<'a>) + 'd,
) {
    // Avoid codegen bloat from monomorphization by immediately doing dyn dispatch of `decorate` to
    // the "real" work.
    fn struct_lint_level_impl(
        sess: &'s Session,
        lint: &'static Lint,
        level: Level,
        src: LintLevelSource,
        span: Option<MultiSpan>,
        decorate: Box<dyn for<'b> FnOnce(LintDiagnosticBuilder<'b>) + 'd>,
    ) {
        // Check for future incompatibility lints and issue a stronger warning.
        let lint_id = LintId::of(lint);
        let future_incompatible = lint.future_incompatible;

        let has_future_breakage = future_incompatible.map_or(
            // Default allow lints trigger too often for testing.
            sess.opts.debugging_opts.future_incompat_test && lint.default_level != Level::Allow,
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
            (Level::Warn, Some(span)) => sess.struct_span_warn(span, ""),
            (Level::Warn, None) => sess.struct_warn(""),
            (Level::ForceWarn, Some(span)) => sess.struct_span_force_warn(span, ""),
            (Level::ForceWarn, None) => sess.struct_force_warn(""),
            (Level::Deny | Level::Forbid, Some(span)) => sess.struct_span_err(span, ""),
            (Level::Deny | Level::Forbid, None) => sess.struct_err(""),
        };

        // If this code originates in a foreign macro, aka something that this crate
        // did not itself author, then it's likely that there's nothing this crate
        // can do about it. We probably want to skip the lint entirely.
        if err.span.primary_spans().iter().any(|s| in_external_macro(sess, *s)) {
            // Any suggestions made here are likely to be incorrect, so anything we
            // emit shouldn't be automatically fixed by rustfix.
            err.allow_suggestions(false);

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

        let name = lint.name_lower();
        match src {
            LintLevelSource::Default => {
                sess.diag_note_once(
                    &mut err,
                    DiagnosticMessageId::from(lint),
                    &format!("`#[{}({})]` on by default", level.as_str(), name),
                );
            }
            LintLevelSource::CommandLine(lint_flag_val, orig_level) => {
                let flag = match orig_level {
                    Level::Warn => "-W",
                    Level::Deny => "-D",
                    Level::Forbid => "-F",
                    Level::Allow => "-A",
                    Level::ForceWarn => "--force-warn",
                };
                let hyphen_case_lint_name = name.replace("_", "-");
                if lint_flag_val.as_str() == name {
                    sess.diag_note_once(
                        &mut err,
                        DiagnosticMessageId::from(lint),
                        &format!(
                            "requested on the command line with `{} {}`",
                            flag, hyphen_case_lint_name
                        ),
                    );
                } else {
                    let hyphen_case_flag_val = lint_flag_val.as_str().replace("_", "-");
                    sess.diag_note_once(
                        &mut err,
                        DiagnosticMessageId::from(lint),
                        &format!(
                            "`{} {}` implied by `{} {}`",
                            flag, hyphen_case_lint_name, flag, hyphen_case_flag_val
                        ),
                    );
                }
            }
            LintLevelSource::Node(lint_attr_name, src, reason) => {
                if let Some(rationale) = reason {
                    err.note(&rationale.as_str());
                }
                sess.diag_span_note_once(
                    &mut err,
                    DiagnosticMessageId::from(lint),
                    src,
                    "the lint level is defined here",
                );
                if lint_attr_name.as_str() != name {
                    let level_str = level.as_str();
                    sess.diag_note_once(
                        &mut err,
                        DiagnosticMessageId::from(lint),
                        &format!(
                            "`#[{}({})]` implied by `#[{}({})]`",
                            level_str, name, level_str, lint_attr_name
                        ),
                    );
                }
            }
        }

        let is_force_warn = matches!(level, Level::ForceWarn);
        err.code(DiagnosticId::Lint { name, has_future_breakage, is_force_warn });

        if let Some(future_incompatible) = future_incompatible {
            let explanation = if lint_id == LintId::of(builtin::UNSTABLE_NAME_COLLISIONS) {
                "once this associated item is added to the standard library, the ambiguity may \
                 cause an error or change in behavior!"
                    .to_owned()
            } else if lint_id == LintId::of(builtin::MUTABLE_BORROW_RESERVATION_CONFLICT) {
                "this borrowing pattern was not meant to be accepted, and may become a hard error \
                 in the future"
                    .to_owned()
            } else if let FutureIncompatibilityReason::EditionError(edition) =
                future_incompatible.reason
            {
                let current_edition = sess.edition();
                format!(
                    "this is accepted in the current edition (Rust {}) but is a hard error in Rust {}!",
                    current_edition, edition
                )
            } else if let FutureIncompatibilityReason::EditionSemanticsChange(edition) =
                future_incompatible.reason
            {
                format!("this changes meaning in Rust {}", edition)
            } else {
                "this was previously accepted by the compiler but is being phased out; \
                 it will become a hard error in a future release!"
                    .to_owned()
            };
            if future_incompatible.explain_reason {
                err.warn(&explanation);
            }
            if !future_incompatible.reference.is_empty() {
                let citation =
                    format!("for more information, see {}", future_incompatible.reference);
                err.note(&citation);
            }
        }

        // Finally, run `decorate`. This function is also responsible for emitting the diagnostic.
        decorate(LintDiagnosticBuilder::new(err));
    }
    struct_lint_level_impl(sess, lint, level, src, span, Box::new(decorate))
}

/// Returns whether `span` originates in a foreign crate's external macro.
///
/// This is used to test whether a lint should not even begin to figure out whether it should
/// be reported on the current node.
pub fn in_external_macro(sess: &Session, span: Span) -> bool {
    let expn_data = span.ctxt().outer_expn_data();
    match expn_data.kind {
        ExpnKind::Inlined | ExpnKind::Root | ExpnKind::Desugaring(DesugaringKind::ForLoop(_)) => {
            false
        }
        ExpnKind::AstPass(_) | ExpnKind::Desugaring(_) => true, // well, it's "external"
        ExpnKind::Macro(MacroKind::Bang, _) => {
            // Dummy span for the `def_site` means it's an external macro.
            expn_data.def_site.is_dummy() || sess.source_map().is_imported(expn_data.def_site)
        }
        ExpnKind::Macro { .. } => true, // definitely a plugin
    }
}
