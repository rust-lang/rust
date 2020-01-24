use std::cmp;

use crate::ich::StableHashingContext;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_errors::{pluralize, Applicability, DiagnosticBuilder, DiagnosticId};
use rustc_hir::HirId;
pub use rustc_session::lint::{builtin, Level, Lint, LintId, LintPass};
use rustc_session::{DiagnosticMessageId, Session};
use rustc_span::hygiene::MacroKind;
use rustc_span::source_map::{DesugaringKind, ExpnKind, MultiSpan};
use rustc_span::{Span, Symbol};

/// How a lint level was set.
#[derive(Clone, Copy, PartialEq, Eq, HashStable)]
pub enum LintSource {
    /// Lint is at the default level as declared
    /// in rustc or a plugin.
    Default,

    /// Lint level was set by an attribute.
    Node(Symbol, Option<Level>, Span, Option<Symbol> /* RFC 2383 reason */),

    /// Lint level was set by a command-line flag.
    CommandLine(Symbol, Option<Level>),
}

pub type LevelSource = (Level, LintSource);

pub struct LintLevelSets {
    pub list: Vec<LintSet>,
    pub lint_cap: Level,
}

pub enum LintSet {
    CommandLine {
        // -A,-W,-D flags, a `Symbol` for the flag itself and `Level` for which
        // flag.
        specs: FxHashMap<LintId, LevelSource>,
    },

    Node {
        specs: FxHashMap<LintId, LevelSource>,
        parent: u32,
    },
}

impl LintLevelSets {
    pub fn new() -> Self {
        LintLevelSets { list: Vec::new(), lint_cap: Level::Forbid }
    }

    pub fn get_lint_level(
        &self,
        lint: &'static Lint,
        idx: u32,
        aux: Option<&FxHashMap<LintId, LevelSource>>,
        sess: &Session,
    ) -> LevelSource {
        let (level, mut src) = self.get_lint_id_level(LintId::of(lint), idx, aux);

        // If `level` is none then we actually assume the default level for this
        // lint.
        let mut level = level.unwrap_or_else(|| lint.default_level(sess.edition()));

        // Ensure we don't go below the minimum level of the lint.
        // Note that we allow `--cap-lints` to cap `WARNINGS`,
        // but we will never allow `--cap-lints` to cap the lint itself.
        let warn_level = cmp::max(level, lint.min_level);

        // If we're about to issue a warning, check at the last minute for any
        // directives against the warnings "lint". If, for example, there's an
        // `allow(warnings)` in scope then we want to respect that instead.
        if warn_level == Level::Warn {
            let (warnings_level, warnings_src) =
                self.get_lint_id_level(LintId::of(builtin::WARNINGS), idx, aux);
            if let Some(configured_warning_level) = warnings_level {
                if configured_warning_level != Level::Warn {
                    let orig_level = Some(level);
                    level = configured_warning_level;
                    src = match warnings_src {
                        LintSource::CommandLine(s, _) => LintSource::CommandLine(s, orig_level),
                        LintSource::Node(n, _, s, r) => LintSource::Node(n, orig_level, s, r),
                        other => other,
                    };
                }
            }
        }

        // Ensure that we never exceed the `--cap-lints` argument.
        level = cmp::min(level, self.lint_cap);

        if let Some(driver_level) = sess.driver_lint_caps.get(&LintId::of(lint)) {
            // Ensure that we never exceed driver level.
            level = cmp::min(*driver_level, level);
        }

        return (level, src);
    }

    pub fn get_lint_id_level(
        &self,
        id: LintId,
        mut idx: u32,
        aux: Option<&FxHashMap<LintId, LevelSource>>,
    ) -> (Option<Level>, LintSource) {
        if let Some(specs) = aux {
            if let Some(&(level, src)) = specs.get(&id) {
                return (Some(level), src);
            }
        }
        loop {
            match self.list[idx as usize] {
                LintSet::CommandLine { ref specs } => {
                    if let Some(&(level, src)) = specs.get(&id) {
                        return (Some(level), src);
                    }
                    return (None, LintSource::Default);
                }
                LintSet::Node { ref specs, parent } => {
                    if let Some(&(level, src)) = specs.get(&id) {
                        return (Some(level), src);
                    }
                    idx = parent;
                }
            }
        }
    }
}

pub struct LintLevelMap {
    pub sets: LintLevelSets,
    pub id_to_set: FxHashMap<HirId, u32>,
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
    ) -> Option<LevelSource> {
        self.id_to_set.get(&id).map(|idx| self.sets.get_lint_level(lint, *idx, None, session))
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for LintLevelMap {
    #[inline]
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        let LintLevelMap { ref sets, ref id_to_set } = *self;

        id_to_set.hash_stable(hcx, hasher);

        let LintLevelSets { ref list, lint_cap } = *sets;

        lint_cap.hash_stable(hcx, hasher);

        hcx.while_hashing_spans(true, |hcx| {
            list.len().hash_stable(hcx, hasher);

            // We are working under the assumption here that the list of
            // lint-sets is built in a deterministic order.
            for lint_set in list {
                ::std::mem::discriminant(lint_set).hash_stable(hcx, hasher);

                match *lint_set {
                    LintSet::CommandLine { ref specs } => {
                        specs.hash_stable(hcx, hasher);
                    }
                    LintSet::Node { ref specs, parent } => {
                        specs.hash_stable(hcx, hasher);
                        parent.hash_stable(hcx, hasher);
                    }
                }
            }
        })
    }
}

fn hyphenate(s: &str) -> String {
    s.replace("_", "-")
}

pub fn struct_lint_level<'a>(
    sess: &'a Session,
    lint: &'static Lint,
    level: Level,
    src: LintSource,
    span: Option<MultiSpan>,
    msg: &str,
) -> DiagnosticBuilder<'a> {
    // Pick the highest level of the given one and the minimum.
    // The effect of this is that if e.g. `min_level == Warn` and
    // you have `#[allow({lint.name})]` then a warning will still
    // be emitted.
    let min_level = lint.min_level;
    let level = cmp::max(orig_level, min_level);

    let mut err = match (level, span) {
        (Level::Allow, _) => return sess.diagnostic().struct_dummy(),
        (Level::Warn, Some(span)) => sess.struct_span_warn(span, msg),
        (Level::Warn, None) => sess.struct_warn(msg),
        (Level::Deny, Some(span)) | (Level::Forbid, Some(span)) => sess.struct_span_err(span, msg),
        (Level::Deny, None) | (Level::Forbid, None) => sess.struct_err(msg),
    };

    // Check for future incompatibility lints and issue a stronger warning.
    let lint_id = LintId::of(lint);
    let future_incompatible = lint.future_incompatible;

    // If this code originates in a foreign macro, aka something that this crate
    // did not itself author, then it's likely that there's nothing this crate
    // can do about it. We probably want to skip the lint entirely.
    if err.span.primary_spans().iter().any(|s| in_external_macro(sess, *s)) {
        // Any suggestions made here are likely to be incorrect, so anything we
        // emit shouldn't be automatically fixed by rustfix.
        err.allow_suggestions(false);

        // If this is a future incompatible lint it'll become a hard error, so
        // we have to emit *something*. Also allow lints to whitelist themselves
        // on a case-by-case basis for emission in a foreign macro.
        if future_incompatible.is_none() && !lint.report_in_external_macro {
            err.cancel();
            // Don't continue further, since we don't want to have
            // `diag_span_note_once` called for a diagnostic that isn't emitted.
            return err;
        }
    }

    let name = lint.name_lower();
    let diag_msg_id = DiagnosticMessageId::from(lint);
    let pre_warn_level = match src {
        LintSource::Default => {
            let msg = &format!("#[{}({})] on by default", orig_level.as_str(), name);
            sess.diag_note_once(&mut err, diag_msg_id, msg);
            None
        }
        LintSource::CommandLine(lint_flag_val) => {
            let flag = orig_level.level_to_flag();
            let lint_name = hyphenate(&name);
            let msg = if lint_flag_val.as_str() == name {
                format!("requested on the command line with `{} {}`", flag, lint_name)
            } else {
                let flag_val = hyphenate(&lint_flag_val.as_str());
                format!("`{} {}` implied by `{} {}`", flag, lint_name, flag, flag_val)
            };
            sess.diag_note_once(&mut err, diag_msg_id, &msg);
            pre_warn_level
        }
        LintSource::Node(lint_attr_name, src, reason) => {
            if orig_level >= level || pre_warn_level.is_some() {
                if let Some(rationale) = reason {
                    err.note(&rationale.as_str());
                }
            }

            sess.diag_span_note_once(&mut err, diag_msg_id, src, "lint level defined here");
            if lint_attr_name.as_str() != name {
                let level_str = orig_level.as_str();
                let msg = format!(
                    "#[{}({})] implied by #[{}({})]",
                    level_str, name, level_str, lint_attr_name
                );
                sess.diag_note_once(&mut err, diag_msg_id, &msg);
            }

            pre_warn_level
        }
    };

    // Highlight the minimum as cause of the lint iff it was raised due to the minimum.
    let orig_level = pre_warn_level.map(|pwl| cmp::min(pwl, orig_level)).unwrap_or(orig_level);
    if orig_level < min_level {
        let min_msg = format!("#[{}({})] is the minimum lint level", min_level.as_str(), name);
        let rem_msg = format!("the lint level cannot be reduced to `{}`", orig_level.as_str());
        sess.diag_note_once(&mut err, diag_msg_id, &min_msg);
        sess.diag_note_once(&mut err, diag_msg_id, &rem_msg)
    }

    err.code(DiagnosticId::Lint(name));

    check_future_compatibility(sess, lint, &mut err, Option::<&str>::None);

    return err;
}

/// Check for future incompatibility lints and issue a stronger warning.
pub fn check_future_compatibility<'a>(
    sess: &'a Session,
    lint: &'static Lint,
    err: &mut DiagnosticBuilder<'_>,
    name: Option<impl fmt::Display>,
) {
    // Check for future incompatibility lints and issue a stronger warning.
    let lints = sess.lint_store.borrow();
    let lint_id = LintId::of(lint);
    let future_incompatible = lints.future_incompatible(lint_id);
    if let Some(future_incompatible) = future_incompatible {
        if lint_id == LintId::of(crate::lint::builtin::UNSTABLE_NAME_COLLISIONS) {
            err.warn(
                "once this method is added to the standard library, \
                      the ambiguity may cause an error or change in behavior!",
            );
        } else if lint_id == LintId::of(builtin::MUTABLE_BORROW_RESERVATION_CONFLICT) {
            err.warn(
                "this borrowing pattern was not meant to be accepted, \
                      and may become a hard error in the future",
            );
        } else {
            let previously_msg = if let Some(n) = name {
                format!("`{}` was previously accepted by the compiler but is being phased out", n)
            } else {
                format!("this was previously accepted by the compiler but is being phased out")
            };
            err.warn(&previously_msg);

            let hard_err_msg = if let Some(edition) = future_incompatible.edition {
                format!("it will become a hard error in the {} edition!", edition)
            } else {
                format!("it will become a hard error in a future release!")
            };
            err.warn(&hard_err_msg);
        }

        err.note(&format!("for more information, see {}", future_incompatible.reference));
    }

    // If this code originates in a foreign macro, aka something that this crate
    // did not itself author, then it's likely that there's nothing this crate
    // can do about it. We probably want to skip the lint entirely.
    if err.span.primary_spans().iter().any(|s| in_external_macro(sess, *s)) {
        // Any suggestions made here are likely to be incorrect, so anything we
        // emit shouldn't be automatically fixed by rustfix.
        err.allow_suggestions(false);

        // If this is a future incompatible lint it'll become a hard error, so
        // we have to emit *something*. Also allow lints to whitelist themselves
        // on a case-by-case basis for emission in a foreign macro.
        if future_incompatible.is_none() && !lint.report_in_external_macro {
            err.cancel()
        }
    }
}

/// Returns whether `span` originates in a foreign crate's external macro.
///
/// This is used to test whether a lint should not even begin to figure out whether it should
/// be reported on the current node.
pub fn in_external_macro(sess: &Session, span: Span) -> bool {
    let expn_data = span.ctxt().outer_expn_data();
    match expn_data.kind {
        ExpnKind::Root | ExpnKind::Desugaring(DesugaringKind::ForLoop) => false,
        ExpnKind::AstPass(_) | ExpnKind::Desugaring(_) => true, // well, it's "external"
        ExpnKind::Macro(MacroKind::Bang, _) => {
            if expn_data.def_site.is_dummy() {
                // Dummy span for the `def_site` means it's an external macro.
                return true;
            }
            match sess.source_map().span_to_snippet(expn_data.def_site) {
                Ok(code) => !code.starts_with("macro_rules"),
                // No snippet means external macro or compiler-builtin expansion.
                Err(_) => true,
            }
        }
        ExpnKind::Macro(..) => true, // definitely a plugin
    }
}

pub fn add_elided_lifetime_in_path_suggestion(
    sess: &Session,
    db: &mut DiagnosticBuilder<'_>,
    n: usize,
    path_span: Span,
    incl_angl_brckt: bool,
    insertion_span: Span,
    anon_lts: String,
) {
    let (replace_span, suggestion) = if incl_angl_brckt {
        (insertion_span, anon_lts)
    } else {
        // When possible, prefer a suggestion that replaces the whole
        // `Path<T>` expression with `Path<'_, T>`, rather than inserting `'_, `
        // at a point (which makes for an ugly/confusing label)
        if let Ok(snippet) = sess.source_map().span_to_snippet(path_span) {
            // But our spans can get out of whack due to macros; if the place we think
            // we want to insert `'_` isn't even within the path expression's span, we
            // should bail out of making any suggestion rather than panicking on a
            // subtract-with-overflow or string-slice-out-out-bounds (!)
            // FIXME: can we do better?
            if insertion_span.lo().0 < path_span.lo().0 {
                return;
            }
            let insertion_index = (insertion_span.lo().0 - path_span.lo().0) as usize;
            if insertion_index > snippet.len() {
                return;
            }
            let (before, after) = snippet.split_at(insertion_index);
            (path_span, format!("{}{}{}", before, anon_lts, after))
        } else {
            (insertion_span, anon_lts)
        }
    };
    db.span_suggestion(
        replace_span,
        &format!("indicate the anonymous lifetime{}", pluralize!(n)),
        suggestion,
        Applicability::MachineApplicable,
    );
}
