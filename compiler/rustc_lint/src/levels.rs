use crate::context::{CheckLintNameResult, LintStore};
use crate::late::unerased_lint_store;
use rustc_ast as ast;
use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{struct_span_err, Applicability, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::{intravisit, HirId, CRATE_HIR_ID};
use rustc_middle::hir::map::Map;
use rustc_middle::lint::LevelAndSource;
use rustc_middle::lint::LintDiagnosticBuilder;
use rustc_middle::lint::{
    struct_lint_level, LintLevelMap, LintLevelSets, LintLevelSource, LintSet, LintStackIndex,
    COMMAND_LINE,
};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::lint::{
    builtin::{self, FORBIDDEN_LINT_GROUPS},
    Level, Lint, LintId,
};
use rustc_session::parse::feature_err;
use rustc_session::Session;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::{source_map::MultiSpan, Span, DUMMY_SP};
use tracing::debug;

fn lint_levels(tcx: TyCtxt<'_>, (): ()) -> LintLevelMap {
    let store = unerased_lint_store(tcx);
    let crate_attrs = tcx.hir().attrs(CRATE_HIR_ID);
    let levels = LintLevelsBuilder::new(tcx.sess, false, &store, crate_attrs);
    let mut builder = LintLevelMapBuilder { levels, tcx, store };
    let krate = tcx.hir().krate();

    builder.levels.id_to_set.reserve(krate.owners.len() + 1);

    let push = builder.levels.push(tcx.hir().attrs(hir::CRATE_HIR_ID), &store, true);
    builder.levels.register_id(hir::CRATE_HIR_ID);
    tcx.hir().walk_toplevel_module(&mut builder);
    builder.levels.pop(push);

    builder.levels.build_map()
}

pub struct LintLevelsBuilder<'s> {
    sess: &'s Session,
    sets: LintLevelSets,
    id_to_set: FxHashMap<HirId, LintStackIndex>,
    cur: LintStackIndex,
    warn_about_weird_lints: bool,
    store: &'s LintStore,
    crate_attrs: &'s [ast::Attribute],
}

pub struct BuilderPush {
    prev: LintStackIndex,
    pub changed: bool,
}

impl<'s> LintLevelsBuilder<'s> {
    pub fn new(
        sess: &'s Session,
        warn_about_weird_lints: bool,
        store: &'s LintStore,
        crate_attrs: &'s [ast::Attribute],
    ) -> Self {
        let mut builder = LintLevelsBuilder {
            sess,
            sets: LintLevelSets::new(),
            cur: COMMAND_LINE,
            id_to_set: Default::default(),
            warn_about_weird_lints,
            store,
            crate_attrs,
        };
        builder.process_command_line(sess, store);
        assert_eq!(builder.sets.list.len(), 1);
        builder
    }

    fn process_command_line(&mut self, sess: &Session, store: &LintStore) {
        let mut specs = FxHashMap::default();
        self.sets.lint_cap = sess.opts.lint_cap.unwrap_or(Level::Forbid);

        for &(ref lint_name, level) in &sess.opts.lint_opts {
            store.check_lint_name_cmdline(sess, &lint_name, level, self.crate_attrs);
            let orig_level = level;
            let lint_flag_val = Symbol::intern(lint_name);

            let ids = match store.find_lints(&lint_name) {
                Ok(ids) => ids,
                Err(_) => continue, // errors handled in check_lint_name_cmdline above
            };
            for id in ids {
                // ForceWarn and Forbid cannot be overriden
                if let Some((Level::ForceWarn | Level::Forbid, _)) = specs.get(&id) {
                    continue;
                }

                self.check_gated_lint(id, DUMMY_SP);
                let src = LintLevelSource::CommandLine(lint_flag_val, orig_level);
                specs.insert(id, (level, src));
            }
        }

        self.cur = self.sets.list.push(LintSet { specs, parent: COMMAND_LINE });
    }

    /// Attempts to insert the `id` to `level_src` map entry. If unsuccessful
    /// (e.g. if a forbid was already inserted on the same scope), then emits a
    /// diagnostic with no change to `specs`.
    fn insert_spec(
        &mut self,
        specs: &mut FxHashMap<LintId, LevelAndSource>,
        id: LintId,
        (level, src): LevelAndSource,
    ) {
        let (old_level, old_src) =
            self.sets.get_lint_level(id.lint, self.cur, Some(&specs), &self.sess);
        // Setting to a non-forbid level is an error if the lint previously had
        // a forbid level. Note that this is not necessarily true even with a
        // `#[forbid(..)]` attribute present, as that is overriden by `--cap-lints`.
        //
        // This means that this only errors if we're truly lowering the lint
        // level from forbid.
        if level != Level::Forbid {
            if let Level::Forbid = old_level {
                // Backwards compatibility check:
                //
                // We used to not consider `forbid(lint_group)`
                // as preventing `allow(lint)` for some lint `lint` in
                // `lint_group`. For now, issue a future-compatibility
                // warning for this case.
                let id_name = id.lint.name_lower();
                let fcw_warning = match old_src {
                    LintLevelSource::Default => false,
                    LintLevelSource::Node(symbol, _, _) => self.store.is_lint_group(symbol),
                    LintLevelSource::CommandLine(symbol, _) => self.store.is_lint_group(symbol),
                };
                debug!(
                    "fcw_warning={:?}, specs.get(&id) = {:?}, old_src={:?}, id_name={:?}",
                    fcw_warning, specs, old_src, id_name
                );

                let decorate_diag_builder = |mut diag_builder: DiagnosticBuilder<'_>| {
                    diag_builder.span_label(src.span(), "overruled by previous forbid");
                    match old_src {
                        LintLevelSource::Default => {
                            diag_builder.note(&format!(
                                "`forbid` lint level is the default for {}",
                                id.to_string()
                            ));
                        }
                        LintLevelSource::Node(_, forbid_source_span, reason) => {
                            diag_builder.span_label(forbid_source_span, "`forbid` level set here");
                            if let Some(rationale) = reason {
                                diag_builder.note(&rationale.as_str());
                            }
                        }
                        LintLevelSource::CommandLine(_, _) => {
                            diag_builder.note("`forbid` lint level was set on command line");
                        }
                    }
                    diag_builder.emit();
                };
                if !fcw_warning {
                    let diag_builder = struct_span_err!(
                        self.sess,
                        src.span(),
                        E0453,
                        "{}({}) incompatible with previous forbid",
                        level.as_str(),
                        src.name(),
                    );
                    decorate_diag_builder(diag_builder);
                } else {
                    self.struct_lint(
                        FORBIDDEN_LINT_GROUPS,
                        Some(src.span().into()),
                        |diag_builder| {
                            let diag_builder = diag_builder.build(&format!(
                                "{}({}) incompatible with previous forbid",
                                level.as_str(),
                                src.name(),
                            ));
                            decorate_diag_builder(diag_builder);
                        },
                    );
                }

                // Retain the forbid lint level, unless we are
                // issuing a FCW. In the FCW case, we want to
                // respect the new setting.
                if !fcw_warning {
                    return;
                }
            }
        }
        if let Level::ForceWarn = old_level {
            specs.insert(id, (old_level, old_src));
        } else {
            specs.insert(id, (level, src));
        }
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

            let mut metas = match attr.meta_item_list() {
                Some(x) => x,
                None => continue,
            };

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
                            // found reason, reslice meta list to exclude it
                            metas.pop().unwrap();
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
                let sp = li.span();
                let mut meta_item = match li {
                    ast::NestedMetaItem::MetaItem(meta_item) if meta_item.is_word() => meta_item,
                    _ => {
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
                let tool_ident = if meta_item.path.segments.len() > 1 {
                    Some(meta_item.path.segments.remove(0).ident)
                } else {
                    None
                };
                let tool_name = tool_ident.map(|ident| ident.name);
                let name = pprust::path_to_string(&meta_item.path);
                let lint_result = store.check_lint_name(sess, &name, tool_name, self.crate_attrs);
                match &lint_result {
                    CheckLintNameResult::Ok(ids) => {
                        let src = LintLevelSource::Node(
                            meta_item.path.segments.last().expect("empty lint name").ident.name,
                            sp,
                            reason,
                        );
                        for &id in *ids {
                            self.check_gated_lint(id, attr.span);
                            self.insert_spec(&mut specs, id, (level, src));
                        }
                    }

                    CheckLintNameResult::Tool(result) => {
                        match *result {
                            Ok(ids) => {
                                let complete_name =
                                    &format!("{}::{}", tool_ident.unwrap().name, name);
                                let src = LintLevelSource::Node(
                                    Symbol::intern(complete_name),
                                    sp,
                                    reason,
                                );
                                for id in ids {
                                    self.insert_spec(&mut specs, *id, (level, src));
                                }
                            }
                            Err((Some(ids), ref new_lint_name)) => {
                                let lint = builtin::RENAMED_AND_REMOVED_LINTS;
                                let (lvl, src) =
                                    self.sets.get_lint_level(lint, self.cur, Some(&specs), &sess);
                                struct_lint_level(
                                    self.sess,
                                    lint,
                                    lvl,
                                    src,
                                    Some(sp.into()),
                                    |lint| {
                                        let msg = format!(
                                            "lint name `{}` is deprecated \
                                             and may not have an effect in the future.",
                                            name
                                        );
                                        lint.build(&msg)
                                            .span_suggestion(
                                                sp,
                                                "change it to",
                                                new_lint_name.to_string(),
                                                Applicability::MachineApplicable,
                                            )
                                            .emit();
                                    },
                                );

                                let src = LintLevelSource::Node(
                                    Symbol::intern(&new_lint_name),
                                    sp,
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

                    &CheckLintNameResult::NoTool => {
                        let mut err = struct_span_err!(
                            sess,
                            tool_ident.map_or(DUMMY_SP, |ident| ident.span),
                            E0710,
                            "unknown tool name `{}` found in scoped lint: `{}::{}`",
                            tool_name.unwrap(),
                            tool_name.unwrap(),
                            pprust::path_to_string(&meta_item.path),
                        );
                        if sess.is_nightly_build() {
                            err.help(&format!(
                                "add `#![register_tool({})]` to the crate root",
                                tool_name.unwrap()
                            ));
                        }
                        err.emit();
                        continue;
                    }

                    _ if !self.warn_about_weird_lints => {}

                    CheckLintNameResult::Warning(msg, renamed) => {
                        let lint = builtin::RENAMED_AND_REMOVED_LINTS;
                        let (renamed_lint_level, src) =
                            self.sets.get_lint_level(lint, self.cur, Some(&specs), &sess);
                        struct_lint_level(
                            self.sess,
                            lint,
                            renamed_lint_level,
                            src,
                            Some(sp.into()),
                            |lint| {
                                let mut err = lint.build(&msg);
                                if let Some(new_name) = &renamed {
                                    err.span_suggestion(
                                        sp,
                                        "use the new name",
                                        new_name.to_string(),
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
                        struct_lint_level(self.sess, lint, level, src, Some(sp.into()), |lint| {
                            let name = if let Some(tool_ident) = tool_ident {
                                format!("{}::{}", tool_ident.name, name)
                            } else {
                                name.to_string()
                            };
                            let mut db = lint.build(&format!("unknown lint: `{}`", name));
                            if let Some(suggestion) = suggestion {
                                db.span_suggestion(
                                    sp,
                                    "did you mean",
                                    suggestion.to_string(),
                                    Applicability::MachineApplicable,
                                );
                            }
                            db.emit();
                        });
                    }
                }
                // If this lint was renamed, apply the new lint instead of ignoring the attribute.
                // This happens outside of the match because the new lint should be applied even if
                // we don't warn about the name change.
                if let CheckLintNameResult::Warning(_, Some(new_name)) = lint_result {
                    // Ignore any errors or warnings that happen because the new name is inaccurate
                    // NOTE: `new_name` already includes the tool name, so we don't have to add it again.
                    if let CheckLintNameResult::Ok(ids) =
                        store.check_lint_name(sess, &new_name, None, self.crate_attrs)
                    {
                        let src = LintLevelSource::Node(Symbol::intern(&new_name), sp, reason);
                        for &id in ids {
                            self.check_gated_lint(id, attr.span);
                            self.insert_spec(&mut specs, id, (level, src));
                        }
                    } else {
                        panic!("renamed lint does not exist: {}", new_name);
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
                    LintLevelSource::Node(name, span, _) => (name, span),
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
            self.cur = self.sets.list.push(LintSet { specs, parent: prev });
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
    pub fn lint_level(&self, lint: &'static Lint) -> (Level, LintLevelSource) {
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

    pub fn build_map(self) -> LintLevelMap {
        LintLevelMap { sets: self.sets, id_to_set: self.id_to_set }
    }
}

pub fn is_known_lint_tool(m_item: Symbol, sess: &Session, attrs: &[ast::Attribute]) -> bool {
    if [sym::clippy, sym::rustc, sym::rustdoc].contains(&m_item) {
        return true;
    }
    // Look for registered tools
    // NOTE: does no error handling; error handling is done by rustc_resolve.
    sess.filter_by_name(attrs, sym::register_tool)
        .filter_map(|attr| attr.meta_item_list())
        .flatten()
        .filter_map(|nested_meta| nested_meta.ident())
        .map(|ident| ident.name)
        .any(|name| name == m_item)
}

struct LintLevelMapBuilder<'a, 'tcx> {
    levels: LintLevelsBuilder<'tcx>,
    tcx: TyCtxt<'tcx>,
    store: &'a LintStore,
}

impl LintLevelMapBuilder<'_, '_> {
    fn with_lint_attrs<F>(&mut self, id: hir::HirId, f: F)
    where
        F: FnOnce(&mut Self),
    {
        let is_crate_hir = id == hir::CRATE_HIR_ID;
        let attrs = self.tcx.hir().attrs(id);
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
        self.with_lint_attrs(param.hir_id, |builder| {
            intravisit::walk_param(builder, param);
        });
    }

    fn visit_item(&mut self, it: &'tcx hir::Item<'tcx>) {
        self.with_lint_attrs(it.hir_id(), |builder| {
            intravisit::walk_item(builder, it);
        });
    }

    fn visit_foreign_item(&mut self, it: &'tcx hir::ForeignItem<'tcx>) {
        self.with_lint_attrs(it.hir_id(), |builder| {
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
        self.with_lint_attrs(e.hir_id, |builder| {
            intravisit::walk_expr(builder, e);
        })
    }

    fn visit_field_def(&mut self, s: &'tcx hir::FieldDef<'tcx>) {
        self.with_lint_attrs(s.hir_id, |builder| {
            intravisit::walk_field_def(builder, s);
        })
    }

    fn visit_variant(
        &mut self,
        v: &'tcx hir::Variant<'tcx>,
        g: &'tcx hir::Generics<'tcx>,
        item_id: hir::HirId,
    ) {
        self.with_lint_attrs(v.id, |builder| {
            intravisit::walk_variant(builder, v, g, item_id);
        })
    }

    fn visit_local(&mut self, l: &'tcx hir::Local<'tcx>) {
        self.with_lint_attrs(l.hir_id, |builder| {
            intravisit::walk_local(builder, l);
        })
    }

    fn visit_arm(&mut self, a: &'tcx hir::Arm<'tcx>) {
        self.with_lint_attrs(a.hir_id, |builder| {
            intravisit::walk_arm(builder, a);
        })
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem<'tcx>) {
        self.with_lint_attrs(trait_item.hir_id(), |builder| {
            intravisit::walk_trait_item(builder, trait_item);
        });
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem<'tcx>) {
        self.with_lint_attrs(impl_item.hir_id(), |builder| {
            intravisit::walk_impl_item(builder, impl_item);
        });
    }
}

pub fn provide(providers: &mut Providers) {
    providers.lint_levels = lint_levels;
}
