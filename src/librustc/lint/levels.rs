use std::cmp;

use crate::ich::StableHashingContext;
use crate::lint::context::{CheckLintNameResult, LintStore};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_errors::{struct_span_err, Applicability, DiagnosticBuilder, DiagnosticId};
use rustc_hir::HirId;
use rustc_session::lint::{builtin, Level, Lint, LintId};
use rustc_session::{DiagnosticMessageId, Session};
use rustc_span::hygiene::MacroKind;
use rustc_span::source_map::{DesugaringKind, ExpnKind, MultiSpan};
use rustc_span::symbol::{sym, Symbol};
use rustc_span::Span;
use syntax::ast;
use syntax::attr;
use syntax::print::pprust;
use syntax::sess::feature_err;

use rustc_error_codes::*;

/// How a lint level was set.
#[derive(Clone, Copy, PartialEq, Eq, HashStable)]
pub enum LintSource {
    /// Lint is at the default level as declared
    /// in rustc or a plugin.
    Default,

    /// Lint level was set by an attribute.
    Node(Symbol, Span, Option<Symbol> /* RFC 2383 reason */),

    /// Lint level was set by a command-line flag.
    CommandLine(Symbol),
}

pub type LevelSource = (Level, LintSource);

pub struct LintLevelSets {
    list: Vec<LintSet>,
    lint_cap: Level,
}

enum LintSet {
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

        // If we're about to issue a warning, check at the last minute for any
        // directives against the warnings "lint". If, for example, there's an
        // `allow(warnings)` in scope then we want to respect that instead.
        if level == Level::Warn {
            let (warnings_level, warnings_src) =
                self.get_lint_id_level(LintId::of(builtin::WARNINGS), idx, aux);
            if let Some(configured_warning_level) = warnings_level {
                if configured_warning_level != Level::Warn {
                    level = configured_warning_level;
                    src = warnings_src;
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

pub struct LintLevelsBuilder<'a> {
    sess: &'a Session,
    sets: LintLevelSets,
    id_to_set: FxHashMap<HirId, u32>,
    cur: u32,
    warn_about_weird_lints: bool,
}

pub struct BuilderPush {
    prev: u32,
    pub changed: bool,
}

impl<'a> LintLevelsBuilder<'a> {
    pub fn new(sess: &'a Session, warn_about_weird_lints: bool, store: &LintStore) -> Self {
        let mut builder = LintLevelsBuilder {
            sess,
            sets: LintLevelSets::new(),
            cur: 0,
            id_to_set: Default::default(),
            warn_about_weird_lints,
        };
        builder.process_command_line(sess, store);
        assert_eq!(builder.sets.list.len(), 1);
        builder
    }

    fn process_command_line(&mut self, sess: &Session, store: &LintStore) {
        let mut specs = FxHashMap::default();
        self.sets.lint_cap = sess.opts.lint_cap.unwrap_or(Level::Forbid);

        for &(ref lint_name, level) in &sess.opts.lint_opts {
            store.check_lint_name_cmdline(sess, &lint_name, level);

            // If the cap is less than this specified level, e.g., if we've got
            // `--cap-lints allow` but we've also got `-D foo` then we ignore
            // this specification as the lint cap will set it to allow anyway.
            let level = cmp::min(level, self.sets.lint_cap);

            let lint_flag_val = Symbol::intern(lint_name);
            let ids = match store.find_lints(&lint_name) {
                Ok(ids) => ids,
                Err(_) => continue, // errors handled in check_lint_name_cmdline above
            };
            for id in ids {
                let src = LintSource::CommandLine(lint_flag_val);
                specs.insert(id, (level, src));
            }
        }

        self.sets.list.push(LintSet::CommandLine { specs });
    }

    /// Pushes a list of AST lint attributes onto this context.
    ///
    /// This function will return a `BuilderPush` object which should be passed
    /// to `pop` when this scope for the attributes provided is exited.
    ///
    /// This function will perform a number of tasks:
    ///
    /// * It'll validate all lint-related attributes in `attrs`
    /// * It'll mark all lint-related attributes as used
    /// * Lint levels will be updated based on the attributes provided
    /// * Lint attributes are validated, e.g., a #[forbid] can't be switched to
    ///   #[allow]
    ///
    /// Don't forget to call `pop`!
    pub fn push(&mut self, attrs: &[ast::Attribute], store: &LintStore) -> BuilderPush {
        let mut specs = FxHashMap::default();
        let sess = self.sess;
        let bad_attr = |span| struct_span_err!(sess, span, E0452, "malformed lint attribute input");
        for attr in attrs {
            let level = match Level::from_symbol(attr.name_or_empty()) {
                None => continue,
                Some(lvl) => lvl,
            };

            let meta = unwrap_or!(attr.meta(), continue);
            attr::mark_used(attr);

            let mut metas = unwrap_or!(meta.meta_item_list(), continue);

            if metas.is_empty() {
                // FIXME (#55112): issue unused-attributes lint for `#[level()]`
                continue;
            }

            // Before processing the lint names, look for a reason (RFC 2383)
            // at the end.
            let mut reason = None;
            let tail_li = &metas[metas.len() - 1];
            if let Some(item) = tail_li.meta_item() {
                match item.kind {
                    ast::MetaItemKind::Word => {} // actual lint names handled later
                    ast::MetaItemKind::NameValue(ref name_value) => {
                        if item.path == sym::reason {
                            // found reason, reslice meta list to exclude it
                            metas = &metas[0..metas.len() - 1];
                            // FIXME (#55112): issue unused-attributes lint if we thereby
                            // don't have any lint names (`#[level(reason = "foo")]`)
                            if let ast::LitKind::Str(rationale, _) = name_value.kind {
                                if !self.sess.features_untracked().lint_reasons {
                                    feature_err(
                                        &self.sess.parse_sess,
                                        sym::lint_reasons,
                                        item.span,
                                        "lint reasons are experimental",
                                    )
                                    .emit();
                                }
                                reason = Some(rationale);
                            } else {
                                bad_attr(name_value.span)
                                    .span_label(name_value.span, "reason must be a string literal")
                                    .emit();
                            }
                        } else {
                            bad_attr(item.span)
                                .span_label(item.span, "bad attribute argument")
                                .emit();
                        }
                    }
                    ast::MetaItemKind::List(_) => {
                        bad_attr(item.span).span_label(item.span, "bad attribute argument").emit();
                    }
                }
            }

            for li in metas {
                let meta_item = match li.meta_item() {
                    Some(meta_item) if meta_item.is_word() => meta_item,
                    _ => {
                        let sp = li.span();
                        let mut err = bad_attr(sp);
                        let mut add_label = true;
                        if let Some(item) = li.meta_item() {
                            if let ast::MetaItemKind::NameValue(_) = item.kind {
                                if item.path == sym::reason {
                                    err.span_label(sp, "reason in lint attribute must come last");
                                    add_label = false;
                                }
                            }
                        }
                        if add_label {
                            err.span_label(sp, "bad attribute argument");
                        }
                        err.emit();
                        continue;
                    }
                };
                let tool_name = if meta_item.path.segments.len() > 1 {
                    let tool_ident = meta_item.path.segments[0].ident;
                    if !attr::is_known_lint_tool(tool_ident) {
                        struct_span_err!(
                            sess,
                            tool_ident.span,
                            E0710,
                            "an unknown tool name found in scoped lint: `{}`",
                            pprust::path_to_string(&meta_item.path),
                        )
                        .emit();
                        continue;
                    }

                    Some(tool_ident.name)
                } else {
                    None
                };
                let name = meta_item.path.segments.last().expect("empty lint name").ident.name;
                match store.check_lint_name(&name.as_str(), tool_name) {
                    CheckLintNameResult::Ok(ids) => {
                        let src = LintSource::Node(name, li.span(), reason);
                        for id in ids {
                            specs.insert(*id, (level, src));
                        }
                    }

                    CheckLintNameResult::Tool(result) => {
                        match result {
                            Ok(ids) => {
                                let complete_name = &format!("{}::{}", tool_name.unwrap(), name);
                                let src = LintSource::Node(
                                    Symbol::intern(complete_name),
                                    li.span(),
                                    reason,
                                );
                                for id in ids {
                                    specs.insert(*id, (level, src));
                                }
                            }
                            Err((Some(ids), new_lint_name)) => {
                                let lint = builtin::RENAMED_AND_REMOVED_LINTS;
                                let (lvl, src) =
                                    self.sets.get_lint_level(lint, self.cur, Some(&specs), &sess);
                                let msg = format!(
                                    "lint name `{}` is deprecated \
                                     and may not have an effect in the future. \
                                     Also `cfg_attr(cargo-clippy)` won't be necessary anymore",
                                    name
                                );
                                struct_lint_level(
                                    self.sess,
                                    lint,
                                    lvl,
                                    src,
                                    Some(li.span().into()),
                                    &msg,
                                )
                                .span_suggestion(
                                    li.span(),
                                    "change it to",
                                    new_lint_name.to_string(),
                                    Applicability::MachineApplicable,
                                )
                                .emit();

                                let src = LintSource::Node(
                                    Symbol::intern(&new_lint_name),
                                    li.span(),
                                    reason,
                                );
                                for id in ids {
                                    specs.insert(*id, (level, src));
                                }
                            }
                            Err((None, _)) => {
                                // If Tool(Err(None, _)) is returned, then either the lint does not
                                // exist in the tool or the code was not compiled with the tool and
                                // therefore the lint was never added to the `LintStore`. To detect
                                // this is the responsibility of the lint tool.
                            }
                        }
                    }

                    _ if !self.warn_about_weird_lints => {}

                    CheckLintNameResult::Warning(msg, renamed) => {
                        let lint = builtin::RENAMED_AND_REMOVED_LINTS;
                        let (level, src) =
                            self.sets.get_lint_level(lint, self.cur, Some(&specs), &sess);
                        let mut err = struct_lint_level(
                            self.sess,
                            lint,
                            level,
                            src,
                            Some(li.span().into()),
                            &msg,
                        );
                        if let Some(new_name) = renamed {
                            err.span_suggestion(
                                li.span(),
                                "use the new name",
                                new_name,
                                Applicability::MachineApplicable,
                            );
                        }
                        err.emit();
                    }
                    CheckLintNameResult::NoLint(suggestion) => {
                        let lint = builtin::UNKNOWN_LINTS;
                        let (level, src) =
                            self.sets.get_lint_level(lint, self.cur, Some(&specs), self.sess);
                        let msg = format!("unknown lint: `{}`", name);
                        let mut db = struct_lint_level(
                            self.sess,
                            lint,
                            level,
                            src,
                            Some(li.span().into()),
                            &msg,
                        );

                        if let Some(suggestion) = suggestion {
                            db.span_suggestion(
                                li.span(),
                                "did you mean",
                                suggestion.to_string(),
                                Applicability::MachineApplicable,
                            );
                        }

                        db.emit();
                    }
                }
            }
        }

        for (id, &(level, ref src)) in specs.iter() {
            if level == Level::Forbid {
                continue;
            }
            let forbid_src = match self.sets.get_lint_id_level(*id, self.cur, None) {
                (Some(Level::Forbid), src) => src,
                _ => continue,
            };
            let forbidden_lint_name = match forbid_src {
                LintSource::Default => id.to_string(),
                LintSource::Node(name, _, _) => name.to_string(),
                LintSource::CommandLine(name) => name.to_string(),
            };
            let (lint_attr_name, lint_attr_span) = match *src {
                LintSource::Node(name, span, _) => (name, span),
                _ => continue,
            };
            let mut diag_builder = struct_span_err!(
                self.sess,
                lint_attr_span,
                E0453,
                "{}({}) overruled by outer forbid({})",
                level.as_str(),
                lint_attr_name,
                forbidden_lint_name
            );
            diag_builder.span_label(lint_attr_span, "overruled by previous forbid");
            match forbid_src {
                LintSource::Default => {}
                LintSource::Node(_, forbid_source_span, reason) => {
                    diag_builder.span_label(forbid_source_span, "`forbid` level set here");
                    if let Some(rationale) = reason {
                        diag_builder.note(&rationale.as_str());
                    }
                }
                LintSource::CommandLine(_) => {
                    diag_builder.note("`forbid` lint level was set on command line");
                }
            }
            diag_builder.emit();
            // don't set a separate error for every lint in the group
            break;
        }

        let prev = self.cur;
        if specs.len() > 0 {
            self.cur = self.sets.list.len() as u32;
            self.sets.list.push(LintSet::Node { specs: specs, parent: prev });
        }

        BuilderPush { prev: prev, changed: prev != self.cur }
    }

    /// Called after `push` when the scope of a set of attributes are exited.
    pub fn pop(&mut self, push: BuilderPush) {
        self.cur = push.prev;
    }

    /// Used to emit a lint-related diagnostic based on the current state of
    /// this lint context.
    pub fn struct_lint(
        &self,
        lint: &'static Lint,
        span: Option<MultiSpan>,
        msg: &str,
    ) -> DiagnosticBuilder<'a> {
        let (level, src) = self.sets.get_lint_level(lint, self.cur, None, self.sess);
        struct_lint_level(self.sess, lint, level, src, span, msg)
    }

    /// Registers the ID provided with the current set of lints stored in
    /// this context.
    pub fn register_id(&mut self, id: HirId) {
        self.id_to_set.insert(id, self.cur);
    }

    pub fn build(self) -> LintLevelSets {
        self.sets
    }

    pub fn build_map(self) -> LintLevelMap {
        LintLevelMap { sets: self.sets, id_to_set: self.id_to_set }
    }
}

pub struct LintLevelMap {
    sets: LintLevelSets,
    id_to_set: FxHashMap<HirId, u32>,
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

pub fn struct_lint_level<'a>(
    sess: &'a Session,
    lint: &'static Lint,
    level: Level,
    src: LintSource,
    span: Option<MultiSpan>,
    msg: &str,
) -> DiagnosticBuilder<'a> {
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
    match src {
        LintSource::Default => {
            sess.diag_note_once(
                &mut err,
                DiagnosticMessageId::from(lint),
                &format!("`#[{}({})]` on by default", level.as_str(), name),
            );
        }
        LintSource::CommandLine(lint_flag_val) => {
            let flag = match level {
                Level::Warn => "-W",
                Level::Deny => "-D",
                Level::Forbid => "-F",
                Level::Allow => panic!(),
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
        LintSource::Node(lint_attr_name, src, reason) => {
            if let Some(rationale) = reason {
                err.note(&rationale.as_str());
            }
            sess.diag_span_note_once(
                &mut err,
                DiagnosticMessageId::from(lint),
                src,
                "lint level defined here",
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

    err.code(DiagnosticId::Lint(name));

    if let Some(future_incompatible) = future_incompatible {
        const STANDARD_MESSAGE: &str = "this was previously accepted by the compiler but is being phased out; \
             it will become a hard error";

        let explanation = if lint_id == LintId::of(builtin::UNSTABLE_NAME_COLLISIONS) {
            "once this method is added to the standard library, \
             the ambiguity may cause an error or change in behavior!"
                .to_owned()
        } else if lint_id == LintId::of(builtin::MUTABLE_BORROW_RESERVATION_CONFLICT) {
            "this borrowing pattern was not meant to be accepted, \
             and may become a hard error in the future"
                .to_owned()
        } else if let Some(edition) = future_incompatible.edition {
            format!("{} in the {} edition!", STANDARD_MESSAGE, edition)
        } else {
            format!("{} in a future release!", STANDARD_MESSAGE)
        };
        let citation = format!("for more information, see {}", future_incompatible.reference);
        err.warn(&explanation);
        err.note(&citation);
    }

    return err;
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
