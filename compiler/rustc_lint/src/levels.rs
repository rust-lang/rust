use crate::context::{CheckLintNameResult, LintStore};
use crate::late::unerased_lint_store;
use rustc_ast as ast;
use rustc_ast::attr;
use rustc_ast::unwrap_or;
use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{struct_span_err, Applicability};
use rustc_hir as hir;
use rustc_hir::def_id::{CrateNum, LOCAL_CRATE};
use rustc_hir::{intravisit, HirId};
use rustc_middle::hir::map::Map;
use rustc_middle::lint::LevelSource;
use rustc_middle::lint::LintDiagnosticBuilder;
use rustc_middle::lint::{struct_lint_level, LintLevelMap, LintLevelSets, LintSet, LintSource};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::lint::{builtin, Level, Lint, LintId};
use rustc_session::parse::feature_err;
use rustc_session::Session;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::{source_map::MultiSpan, Span, DUMMY_SP};

use std::cmp;

fn lint_levels(tcx: TyCtxt<'_>, cnum: CrateNum) -> LintLevelMap {
    assert_eq!(cnum, LOCAL_CRATE);
    let store = unerased_lint_store(tcx);
    let levels = LintLevelsBuilder::new(tcx.sess, false, &store);
    let mut builder = LintLevelMapBuilder { levels, tcx, store };
    let krate = tcx.hir().krate();

    builder.levels.id_to_set.reserve(krate.exported_macros.len() + 1);

    let push = builder.levels.push(&krate.item.attrs, &store, true);
    builder.levels.register_id(hir::CRATE_HIR_ID);
    for macro_def in krate.exported_macros {
        builder.levels.register_id(macro_def.hir_id);
    }
    intravisit::walk_crate(&mut builder, krate);
    builder.levels.pop(push);

    builder.levels.build_map()
}

pub struct LintLevelsBuilder<'s> {
    sess: &'s Session,
    sets: LintLevelSets,
    id_to_set: FxHashMap<HirId, u32>,
    cur: u32,
    warn_about_weird_lints: bool,
}

pub struct BuilderPush {
    prev: u32,
    pub changed: bool,
}

impl<'s> LintLevelsBuilder<'s> {
    pub fn new(sess: &'s Session, warn_about_weird_lints: bool, store: &LintStore) -> Self {
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
            let orig_level = level;

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
                self.check_gated_lint(id, DUMMY_SP);
                let src = LintSource::CommandLine(lint_flag_val, orig_level);
                specs.insert(id, (level, src));
            }
        }

        self.sets.list.push(LintSet::CommandLine { specs });
    }

    /// Attempts to insert the `id` to `level_src` map entry. If unsuccessful
    /// (e.g. if a forbid was already inserted on the same scope), then emits a
    /// diagnostic with no change to `specs`.
    fn insert_spec(
        &mut self,
        specs: &mut FxHashMap<LintId, LevelSource>,
        id: LintId,
        (level, src): LevelSource,
    ) {
        // Setting to a non-forbid level is an error if the lint previously had
        // a forbid level. Note that this is not necessarily true even with a
        // `#[forbid(..)]` attribute present, as that is overriden by `--cap-lints`.
        //
        // This means that this only errors if we're truly lowering the lint
        // level from forbid.
        if level != Level::Forbid {
            if let (Level::Forbid, old_src) =
                self.sets.get_lint_level(id.lint, self.cur, Some(&specs), &self.sess)
            {
                let mut diag_builder = struct_span_err!(
                    self.sess,
                    src.span(),
                    E0453,
                    "{}({}) incompatible with previous forbid",
                    level.as_str(),
                    src.name(),
                );
                diag_builder.span_label(src.span(), "overruled by previous forbid");
                match old_src {
                    LintSource::Default => {
                        diag_builder.note(&format!(
                            "`forbid` lint level is the default for {}",
                            id.to_string()
                        ));
                    }
                    LintSource::Node(_, forbid_source_span, reason) => {
                        diag_builder.span_label(forbid_source_span, "`forbid` level set here");
                        if let Some(rationale) = reason {
                            diag_builder.note(&rationale.as_str());
                        }
                    }
                    LintSource::CommandLine(_, _) => {
                        diag_builder.note("`forbid` lint level was set on command line");
                    }
                }
                diag_builder.emit();

                // Retain the forbid lint level
                return;
            }
        }
        specs.insert(id, (level, src));
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
    /// * Lint attributes are validated, e.g., a `#[forbid]` can't be switched to
    ///   `#[allow]`
    ///
    /// Don't forget to call `pop`!
    pub(crate) fn push(
        &mut self,
        attrs: &[ast::Attribute],
        store: &LintStore,
        is_crate_node: bool,
    ) -> BuilderPush {
        let mut specs = FxHashMap::default();
        let sess = self.sess;
        let bad_attr = |span| struct_span_err!(sess, span, E0452, "malformed lint attribute input");
        for attr in attrs {
            let level = match Level::from_symbol(attr.name_or_empty()) {
                None => continue,
                Some(lvl) => lvl,
            };

            let meta = unwrap_or!(attr.meta(), continue);
            self.sess.mark_attr_used(attr);

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
                        for &id in ids {
                            self.check_gated_lint(id, attr.span);
                            self.insert_spec(&mut specs, id, (level, src));
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
                                    self.insert_spec(&mut specs, *id, (level, src));
                                }
                            }
                            Err((Some(ids), new_lint_name)) => {
                                let lint = builtin::RENAMED_AND_REMOVED_LINTS;
                                let (lvl, src) =
                                    self.sets.get_lint_level(lint, self.cur, Some(&specs), &sess);
                                struct_lint_level(
                                    self.sess,
                                    lint,
                                    lvl,
                                    src,
                                    Some(li.span().into()),
                                    |lint| {
                                        let msg = format!(
                                            "lint name `{}` is deprecated \
                                             and may not have an effect in the future. \
                                             Also `cfg_attr(cargo-clippy)` won't be necessary anymore",
                                            name
                                        );
                                        lint.build(&msg)
                                            .span_suggestion(
                                                li.span(),
                                                "change it to",
                                                new_lint_name.to_string(),
                                                Applicability::MachineApplicable,
                                            )
                                            .emit();
                                    },
                                );

                                let src = LintSource::Node(
                                    Symbol::intern(&new_lint_name),
                                    li.span(),
                                    reason,
                                );
                                for id in ids {
                                    self.insert_spec(&mut specs, *id, (level, src));
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
                        struct_lint_level(
                            self.sess,
                            lint,
                            level,
                            src,
                            Some(li.span().into()),
                            |lint| {
                                let mut err = lint.build(&msg);
                                if let Some(new_name) = renamed {
                                    err.span_suggestion(
                                        li.span(),
                                        "use the new name",
                                        new_name,
                                        Applicability::MachineApplicable,
                                    );
                                }
                                err.emit();
                            },
                        );
                    }
                    CheckLintNameResult::NoLint(suggestion) => {
                        let lint = builtin::UNKNOWN_LINTS;
                        let (level, src) =
                            self.sets.get_lint_level(lint, self.cur, Some(&specs), self.sess);
                        struct_lint_level(
                            self.sess,
                            lint,
                            level,
                            src,
                            Some(li.span().into()),
                            |lint| {
                                let mut db = lint.build(&format!("unknown lint: `{}`", name));
                                if let Some(suggestion) = suggestion {
                                    db.span_suggestion(
                                        li.span(),
                                        "did you mean",
                                        suggestion.to_string(),
                                        Applicability::MachineApplicable,
                                    );
                                }
                                db.emit();
                            },
                        );
                    }
                }
            }
        }

        if !is_crate_node {
            for (id, &(level, ref src)) in specs.iter() {
                if !id.lint.crate_level_only {
                    continue;
                }

                let (lint_attr_name, lint_attr_span) = match *src {
                    LintSource::Node(name, span, _) => (name, span),
                    _ => continue,
                };

                let lint = builtin::UNUSED_ATTRIBUTES;
                let (lint_level, lint_src) =
                    self.sets.get_lint_level(lint, self.cur, Some(&specs), self.sess);
                struct_lint_level(
                    self.sess,
                    lint,
                    lint_level,
                    lint_src,
                    Some(lint_attr_span.into()),
                    |lint| {
                        let mut db = lint.build(&format!(
                            "{}({}) is ignored unless specified at crate level",
                            level.as_str(),
                            lint_attr_name
                        ));
                        db.emit();
                    },
                );
                // don't set a separate error for every lint in the group
                break;
            }
        }

        let prev = self.cur;
        if !specs.is_empty() {
            self.cur = self.sets.list.len() as u32;
            self.sets.list.push(LintSet::Node { specs, parent: prev });
        }

        BuilderPush { prev, changed: prev != self.cur }
    }

    /// Checks if the lint is gated on a feature that is not enabled.
    fn check_gated_lint(&self, lint_id: LintId, span: Span) {
        if let Some(feature) = lint_id.lint.feature_gate {
            if !self.sess.features_untracked().enabled(feature) {
                feature_err(
                    &self.sess.parse_sess,
                    feature,
                    span,
                    &format!("the `{}` lint is unstable", lint_id.lint.name_lower()),
                )
                .emit();
            }
        }
    }

    /// Called after `push` when the scope of a set of attributes are exited.
    pub fn pop(&mut self, push: BuilderPush) {
        self.cur = push.prev;
    }

    /// Find the lint level for a lint.
    pub fn lint_level(&self, lint: &'static Lint) -> (Level, LintSource) {
        self.sets.get_lint_level(lint, self.cur, None, self.sess)
    }

    /// Used to emit a lint-related diagnostic based on the current state of
    /// this lint context.
    pub fn struct_lint(
        &self,
        lint: &'static Lint,
        span: Option<MultiSpan>,
        decorate: impl for<'a> FnOnce(LintDiagnosticBuilder<'a>),
    ) {
        let (level, src) = self.lint_level(lint);
        struct_lint_level(self.sess, lint, level, src, span, decorate)
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

struct LintLevelMapBuilder<'a, 'tcx> {
    levels: LintLevelsBuilder<'tcx>,
    tcx: TyCtxt<'tcx>,
    store: &'a LintStore,
}

impl LintLevelMapBuilder<'_, '_> {
    fn with_lint_attrs<F>(&mut self, id: hir::HirId, attrs: &[ast::Attribute], f: F)
    where
        F: FnOnce(&mut Self),
    {
        let is_crate_hir = id == hir::CRATE_HIR_ID;
        let push = self.levels.push(attrs, self.store, is_crate_hir);
        if push.changed {
            self.levels.register_id(id);
        }
        f(self);
        self.levels.pop(push);
    }
}

impl<'tcx> intravisit::Visitor<'tcx> for LintLevelMapBuilder<'_, 'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
        intravisit::NestedVisitorMap::All(self.tcx.hir())
    }

    fn visit_param(&mut self, param: &'tcx hir::Param<'tcx>) {
        self.with_lint_attrs(param.hir_id, &param.attrs, |builder| {
            intravisit::walk_param(builder, param);
        });
    }

    fn visit_item(&mut self, it: &'tcx hir::Item<'tcx>) {
        self.with_lint_attrs(it.hir_id, &it.attrs, |builder| {
            intravisit::walk_item(builder, it);
        });
    }

    fn visit_foreign_item(&mut self, it: &'tcx hir::ForeignItem<'tcx>) {
        self.with_lint_attrs(it.hir_id, &it.attrs, |builder| {
            intravisit::walk_foreign_item(builder, it);
        })
    }

    fn visit_stmt(&mut self, e: &'tcx hir::Stmt<'tcx>) {
        // We will call `with_lint_attrs` when we walk
        // the `StmtKind`. The outer statement itself doesn't
        // define the lint levels.
        intravisit::walk_stmt(self, e);
    }

    fn visit_expr(&mut self, e: &'tcx hir::Expr<'tcx>) {
        self.with_lint_attrs(e.hir_id, &e.attrs, |builder| {
            intravisit::walk_expr(builder, e);
        })
    }

    fn visit_struct_field(&mut self, s: &'tcx hir::StructField<'tcx>) {
        self.with_lint_attrs(s.hir_id, &s.attrs, |builder| {
            intravisit::walk_struct_field(builder, s);
        })
    }

    fn visit_variant(
        &mut self,
        v: &'tcx hir::Variant<'tcx>,
        g: &'tcx hir::Generics<'tcx>,
        item_id: hir::HirId,
    ) {
        self.with_lint_attrs(v.id, &v.attrs, |builder| {
            intravisit::walk_variant(builder, v, g, item_id);
        })
    }

    fn visit_local(&mut self, l: &'tcx hir::Local<'tcx>) {
        self.with_lint_attrs(l.hir_id, &l.attrs, |builder| {
            intravisit::walk_local(builder, l);
        })
    }

    fn visit_arm(&mut self, a: &'tcx hir::Arm<'tcx>) {
        self.with_lint_attrs(a.hir_id, &a.attrs, |builder| {
            intravisit::walk_arm(builder, a);
        })
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem<'tcx>) {
        self.with_lint_attrs(trait_item.hir_id, &trait_item.attrs, |builder| {
            intravisit::walk_trait_item(builder, trait_item);
        });
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem<'tcx>) {
        self.with_lint_attrs(impl_item.hir_id, &impl_item.attrs, |builder| {
            intravisit::walk_impl_item(builder, impl_item);
        });
    }
}

pub fn provide(providers: &mut Providers) {
    providers.lint_levels = lint_levels;
}
