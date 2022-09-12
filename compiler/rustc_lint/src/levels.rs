use crate::context::{CheckLintNameResult, LintStore};
use crate::late::unerased_lint_store;
use rustc_ast as ast;
use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{Applicability, Diagnostic, LintDiagnosticBuilder, MultiSpan};
use rustc_hir as hir;
use rustc_hir::{intravisit, HirId};
use rustc_middle::hir::nested_filter;
use rustc_middle::lint::{
    struct_lint_level, LevelAndSource, LintExpectation, LintLevelMap, LintLevelSets,
    LintLevelSource, LintSet, LintStackIndex, COMMAND_LINE,
};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{RegisteredTools, TyCtxt};
use rustc_session::lint::{
    builtin::{self, FORBIDDEN_LINT_GROUPS, SINGLE_USE_LIFETIMES, UNFULFILLED_LINT_EXPECTATIONS},
    Level, Lint, LintExpectationId, LintId,
};
use rustc_session::parse::{add_feature_diagnostics, feature_err};
use rustc_session::Session;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::{Span, DUMMY_SP};

use crate::errors::{
    MalformedAttribute, MalformedAttributeSub, OverruledAttribute, OverruledAttributeSub,
    UnknownToolInScopedLint,
};

fn lint_levels(tcx: TyCtxt<'_>, (): ()) -> LintLevelMap {
    let store = unerased_lint_store(tcx);
    let levels =
        LintLevelsBuilder::new(tcx.sess, false, &store, &tcx.resolutions(()).registered_tools);
    let mut builder = LintLevelMapBuilder { levels, tcx };
    let krate = tcx.hir().krate();

    builder.levels.id_to_set.reserve(krate.owners.len() + 1);

    let push =
        builder.levels.push(tcx.hir().attrs(hir::CRATE_HIR_ID), true, Some(hir::CRATE_HIR_ID));

    builder.levels.register_id(hir::CRATE_HIR_ID);
    tcx.hir().walk_toplevel_module(&mut builder);
    builder.levels.pop(push);

    builder.levels.update_unstable_expectation_ids();
    builder.levels.build_map()
}

pub struct LintLevelsBuilder<'s> {
    sess: &'s Session,
    lint_expectations: Vec<(LintExpectationId, LintExpectation)>,
    /// Each expectation has a stable and an unstable identifier. This map
    /// is used to map from unstable to stable [`LintExpectationId`]s.
    expectation_id_map: FxHashMap<LintExpectationId, LintExpectationId>,
    sets: LintLevelSets,
    id_to_set: FxHashMap<HirId, LintStackIndex>,
    cur: LintStackIndex,
    warn_about_weird_lints: bool,
    store: &'s LintStore,
    registered_tools: &'s RegisteredTools,
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
        registered_tools: &'s RegisteredTools,
    ) -> Self {
        let mut builder = LintLevelsBuilder {
            sess,
            lint_expectations: Default::default(),
            expectation_id_map: Default::default(),
            sets: LintLevelSets::new(),
            cur: COMMAND_LINE,
            id_to_set: Default::default(),
            warn_about_weird_lints,
            store,
            registered_tools,
        };
        builder.process_command_line(sess, store);
        assert_eq!(builder.sets.list.len(), 1);
        builder
    }

    pub(crate) fn sess(&self) -> &Session {
        self.sess
    }

    pub(crate) fn lint_store(&self) -> &LintStore {
        self.store
    }

    fn current_specs(&self) -> &FxHashMap<LintId, LevelAndSource> {
        &self.sets.list[self.cur].specs
    }

    fn current_specs_mut(&mut self) -> &mut FxHashMap<LintId, LevelAndSource> {
        &mut self.sets.list[self.cur].specs
    }

    fn process_command_line(&mut self, sess: &Session, store: &LintStore) {
        self.sets.lint_cap = sess.opts.lint_cap.unwrap_or(Level::Forbid);

        self.cur =
            self.sets.list.push(LintSet { specs: FxHashMap::default(), parent: COMMAND_LINE });
        for &(ref lint_name, level) in &sess.opts.lint_opts {
            store.check_lint_name_cmdline(sess, &lint_name, level, self.registered_tools);
            let orig_level = level;
            let lint_flag_val = Symbol::intern(lint_name);

            let Ok(ids) = store.find_lints(&lint_name) else {
                // errors handled in check_lint_name_cmdline above
                continue
            };
            for id in ids {
                // ForceWarn and Forbid cannot be overridden
                if let Some((Level::ForceWarn(_) | Level::Forbid, _)) =
                    self.current_specs().get(&id)
                {
                    continue;
                }

                if self.check_gated_lint(id, DUMMY_SP) {
                    let src = LintLevelSource::CommandLine(lint_flag_val, orig_level);
                    self.current_specs_mut().insert(id, (level, src));
                }
            }
        }
    }

    /// Attempts to insert the `id` to `level_src` map entry. If unsuccessful
    /// (e.g. if a forbid was already inserted on the same scope), then emits a
    /// diagnostic with no change to `specs`.
    fn insert_spec(&mut self, id: LintId, (level, src): LevelAndSource) {
        let (old_level, old_src) =
            self.sets.get_lint_level(id.lint, self.cur, Some(self.current_specs()), &self.sess);
        // Setting to a non-forbid level is an error if the lint previously had
        // a forbid level. Note that this is not necessarily true even with a
        // `#[forbid(..)]` attribute present, as that is overridden by `--cap-lints`.
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
                    fcw_warning,
                    self.current_specs(),
                    old_src,
                    id_name
                );

                let decorate_diag = |diag: &mut Diagnostic| {
                    diag.span_label(src.span(), "overruled by previous forbid");
                    match old_src {
                        LintLevelSource::Default => {
                            diag.note(&format!(
                                "`forbid` lint level is the default for {}",
                                id.to_string()
                            ));
                        }
                        LintLevelSource::Node(_, forbid_source_span, reason) => {
                            diag.span_label(forbid_source_span, "`forbid` level set here");
                            if let Some(rationale) = reason {
                                diag.note(rationale.as_str());
                            }
                        }
                        LintLevelSource::CommandLine(_, _) => {
                            diag.note("`forbid` lint level was set on command line");
                        }
                    }
                };
                if !fcw_warning {
                    self.sess.emit_err(OverruledAttribute {
                        span: src.span(),
                        overruled: src.span(),
                        lint_level: level.as_str().to_string(),
                        lint_source: src.name(),
                        sub: match old_src {
                            LintLevelSource::Default => {
                                OverruledAttributeSub::DefaultSource { id: id.to_string() }
                            }
                            LintLevelSource::Node(_, forbid_source_span, reason) => {
                                OverruledAttributeSub::NodeSource {
                                    span: forbid_source_span,
                                    reason,
                                }
                            }
                            LintLevelSource::CommandLine(_, _) => {
                                OverruledAttributeSub::CommandLineSource
                            }
                        },
                    });
                } else {
                    self.struct_lint(
                        FORBIDDEN_LINT_GROUPS,
                        Some(src.span().into()),
                        |diag_builder| {
                            let mut diag_builder = diag_builder.build(&format!(
                                "{}({}) incompatible with previous forbid",
                                level.as_str(),
                                src.name(),
                            ));
                            decorate_diag(&mut diag_builder);
                            diag_builder.emit();
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

        // The lint `unfulfilled_lint_expectations` can't be expected, as it would suppress itself.
        // Handling expectations of this lint would add additional complexity with little to no
        // benefit. The expect level for this lint will therefore be ignored.
        if let Level::Expect(_) = level && id == LintId::of(UNFULFILLED_LINT_EXPECTATIONS) {
            return;
        }

        match (old_level, level) {
            // If the new level is an expectation store it in `ForceWarn`
            (Level::ForceWarn(_), Level::Expect(expectation_id)) => self
                .current_specs_mut()
                .insert(id, (Level::ForceWarn(Some(expectation_id)), old_src)),
            // Keep `ForceWarn` level but drop the expectation
            (Level::ForceWarn(_), _) => {
                self.current_specs_mut().insert(id, (Level::ForceWarn(None), old_src))
            }
            // Set the lint level as normal
            _ => self.current_specs_mut().insert(id, (level, src)),
        };
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
        is_crate_node: bool,
        source_hir_id: Option<HirId>,
    ) -> BuilderPush {
        let prev = self.cur;
        self.cur = self.sets.list.push(LintSet { specs: FxHashMap::default(), parent: prev });

        let sess = self.sess;
        for (attr_index, attr) in attrs.iter().enumerate() {
            if attr.has_name(sym::automatically_derived) {
                self.current_specs_mut().insert(
                    LintId::of(SINGLE_USE_LIFETIMES),
                    (Level::Allow, LintLevelSource::Default),
                );
                continue;
            }

            let level = match Level::from_attr(attr) {
                None => continue,
                // This is the only lint level with a `LintExpectationId` that can be created from an attribute
                Some(Level::Expect(unstable_id)) if let Some(hir_id) = source_hir_id => {
                    let stable_id = self.create_stable_id(unstable_id, hir_id, attr_index);

                    Level::Expect(stable_id)
                }
                Some(lvl) => lvl,
            };

            let Some(mut metas) = attr.meta_item_list() else {
                continue
            };

            if metas.is_empty() {
                // This emits the unused_attributes lint for `#[level()]`
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
                                sess.emit_err(MalformedAttribute {
                                    span: name_value.span,
                                    sub: MalformedAttributeSub::ReasonMustBeStringLiteral(
                                        name_value.span,
                                    ),
                                });
                            }
                            // found reason, reslice meta list to exclude it
                            metas.pop().unwrap();
                        } else {
                            sess.emit_err(MalformedAttribute {
                                span: item.span,
                                sub: MalformedAttributeSub::BadAttributeArgument(item.span),
                            });
                        }
                    }
                    ast::MetaItemKind::List(_) => {
                        sess.emit_err(MalformedAttribute {
                            span: item.span,
                            sub: MalformedAttributeSub::BadAttributeArgument(item.span),
                        });
                    }
                }
            }

            for (lint_index, li) in metas.iter_mut().enumerate() {
                let level = match level {
                    Level::Expect(mut id) => {
                        id.set_lint_index(Some(lint_index as u16));
                        Level::Expect(id)
                    }
                    level => level,
                };

                let sp = li.span();
                let meta_item = match li {
                    ast::NestedMetaItem::MetaItem(meta_item) if meta_item.is_word() => meta_item,
                    _ => {
                        if let Some(item) = li.meta_item() {
                            if let ast::MetaItemKind::NameValue(_) = item.kind {
                                if item.path == sym::reason {
                                    sess.emit_err(MalformedAttribute {
                                        span: sp,
                                        sub: MalformedAttributeSub::ReasonMustComeLast(sp),
                                    });
                                    continue;
                                }
                            }
                        }
                        sess.emit_err(MalformedAttribute {
                            span: sp,
                            sub: MalformedAttributeSub::BadAttributeArgument(sp),
                        });
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
                let lint_result =
                    self.store.check_lint_name(&name, tool_name, self.registered_tools);
                match &lint_result {
                    CheckLintNameResult::Ok(ids) => {
                        // This checks for instances where the user writes `#[expect(unfulfilled_lint_expectations)]`
                        // in that case we want to avoid overriding the lint level but instead add an expectation that
                        // can't be fulfilled. The lint message will include an explanation, that the
                        // `unfulfilled_lint_expectations` lint can't be expected.
                        if let Level::Expect(expect_id) = level {
                            // The `unfulfilled_lint_expectations` lint is not part of any lint groups. Therefore. we
                            // only need to check the slice if it contains a single lint.
                            let is_unfulfilled_lint_expectations = match ids {
                                [lint] => *lint == LintId::of(UNFULFILLED_LINT_EXPECTATIONS),
                                _ => false,
                            };
                            self.lint_expectations.push((
                                expect_id,
                                LintExpectation::new(
                                    reason,
                                    sp,
                                    is_unfulfilled_lint_expectations,
                                    tool_name,
                                ),
                            ));
                        }
                        let src = LintLevelSource::Node(
                            meta_item.path.segments.last().expect("empty lint name").ident.name,
                            sp,
                            reason,
                        );
                        for &id in *ids {
                            if self.check_gated_lint(id, attr.span) {
                                self.insert_spec(id, (level, src));
                            }
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
                                for &id in ids {
                                    if self.check_gated_lint(id, attr.span) {
                                        self.insert_spec(id, (level, src));
                                    }
                                }
                                if let Level::Expect(expect_id) = level {
                                    self.lint_expectations.push((
                                        expect_id,
                                        LintExpectation::new(reason, sp, false, tool_name),
                                    ));
                                }
                            }
                            Err((Some(ids), ref new_lint_name)) => {
                                let lint = builtin::RENAMED_AND_REMOVED_LINTS;
                                let (lvl, src) = self.sets.get_lint_level(
                                    lint,
                                    self.cur,
                                    Some(self.current_specs()),
                                    &sess,
                                );
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
                                                new_lint_name,
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
                                    self.insert_spec(*id, (level, src));
                                }
                                if let Level::Expect(expect_id) = level {
                                    self.lint_expectations.push((
                                        expect_id,
                                        LintExpectation::new(reason, sp, false, tool_name),
                                    ));
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
                        sess.emit_err(UnknownToolInScopedLint {
                            span: tool_ident.map(|ident| ident.span),
                            tool_name: tool_name.unwrap(),
                            lint_name: pprust::path_to_string(&meta_item.path),
                            is_nightly_build: sess.is_nightly_build().then_some(()),
                        });
                        continue;
                    }

                    _ if !self.warn_about_weird_lints => {}

                    CheckLintNameResult::Warning(msg, renamed) => {
                        let lint = builtin::RENAMED_AND_REMOVED_LINTS;
                        let (renamed_lint_level, src) = self.sets.get_lint_level(
                            lint,
                            self.cur,
                            Some(self.current_specs()),
                            &sess,
                        );
                        struct_lint_level(
                            self.sess,
                            lint,
                            renamed_lint_level,
                            src,
                            Some(sp.into()),
                            |lint| {
                                let mut err = lint.build(msg);
                                if let Some(new_name) = &renamed {
                                    err.span_suggestion(
                                        sp,
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
                        let (level, src) = self.sets.get_lint_level(
                            lint,
                            self.cur,
                            Some(self.current_specs()),
                            self.sess,
                        );
                        struct_lint_level(self.sess, lint, level, src, Some(sp.into()), |lint| {
                            let name = if let Some(tool_ident) = tool_ident {
                                format!("{}::{}", tool_ident.name, name)
                            } else {
                                name.to_string()
                            };
                            let mut db = lint.build(format!("unknown lint: `{}`", name));
                            if let Some(suggestion) = suggestion {
                                db.span_suggestion(
                                    sp,
                                    "did you mean",
                                    suggestion,
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
                        self.store.check_lint_name(&new_name, None, self.registered_tools)
                    {
                        let src = LintLevelSource::Node(Symbol::intern(&new_name), sp, reason);
                        for &id in ids {
                            if self.check_gated_lint(id, attr.span) {
                                self.insert_spec(id, (level, src));
                            }
                        }
                        if let Level::Expect(expect_id) = level {
                            self.lint_expectations.push((
                                expect_id,
                                LintExpectation::new(reason, sp, false, tool_name),
                            ));
                        }
                    } else {
                        panic!("renamed lint does not exist: {}", new_name);
                    }
                }
            }
        }

        if !is_crate_node {
            for (id, &(level, ref src)) in self.current_specs().iter() {
                if !id.lint.crate_level_only {
                    continue;
                }

                let LintLevelSource::Node(lint_attr_name, lint_attr_span, _) = *src else {
                    continue
                };

                let lint = builtin::UNUSED_ATTRIBUTES;
                let (lint_level, lint_src) =
                    self.sets.get_lint_level(lint, self.cur, Some(self.current_specs()), self.sess);
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

        if self.current_specs().is_empty() {
            self.sets.list.pop();
            self.cur = prev;
        }

        BuilderPush { prev, changed: prev != self.cur }
    }

    fn create_stable_id(
        &mut self,
        unstable_id: LintExpectationId,
        hir_id: HirId,
        attr_index: usize,
    ) -> LintExpectationId {
        let stable_id =
            LintExpectationId::Stable { hir_id, attr_index: attr_index as u16, lint_index: None };

        self.expectation_id_map.insert(unstable_id, stable_id);

        stable_id
    }

    /// Checks if the lint is gated on a feature that is not enabled.
    ///
    /// Returns `true` if the lint's feature is enabled.
    fn check_gated_lint(&self, lint_id: LintId, span: Span) -> bool {
        if let Some(feature) = lint_id.lint.feature_gate {
            if !self.sess.features_untracked().enabled(feature) {
                let lint = builtin::UNKNOWN_LINTS;
                let (level, src) = self.lint_level(builtin::UNKNOWN_LINTS);
                struct_lint_level(self.sess, lint, level, src, Some(span.into()), |lint_db| {
                    let mut db =
                        lint_db.build(&format!("unknown lint: `{}`", lint_id.lint.name_lower()));
                    db.note(&format!("the `{}` lint is unstable", lint_id.lint.name_lower(),));
                    add_feature_diagnostics(&mut db, &self.sess.parse_sess, feature);
                    db.emit();
                });
                return false;
            }
        }
        true
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
        decorate: impl for<'a> FnOnce(LintDiagnosticBuilder<'a, ()>),
    ) {
        let (level, src) = self.lint_level(lint);
        struct_lint_level(self.sess, lint, level, src, span, decorate)
    }

    /// Registers the ID provided with the current set of lints stored in
    /// this context.
    pub fn register_id(&mut self, id: HirId) {
        self.id_to_set.insert(id, self.cur);
    }

    fn update_unstable_expectation_ids(&self) {
        self.sess.diagnostic().update_unstable_expectation_id(&self.expectation_id_map);
    }

    pub fn build_map(self) -> LintLevelMap {
        LintLevelMap {
            sets: self.sets,
            id_to_set: self.id_to_set,
            lint_expectations: self.lint_expectations,
        }
    }
}

struct LintLevelMapBuilder<'tcx> {
    levels: LintLevelsBuilder<'tcx>,
    tcx: TyCtxt<'tcx>,
}

impl LintLevelMapBuilder<'_> {
    fn with_lint_attrs<F>(&mut self, id: hir::HirId, f: F)
    where
        F: FnOnce(&mut Self),
    {
        let is_crate_hir = id == hir::CRATE_HIR_ID;
        let attrs = self.tcx.hir().attrs(id);
        let push = self.levels.push(attrs, is_crate_hir, Some(id));

        if push.changed {
            self.levels.register_id(id);
        }
        f(self);
        self.levels.pop(push);
    }
}

impl<'tcx> intravisit::Visitor<'tcx> for LintLevelMapBuilder<'tcx> {
    type NestedFilter = nested_filter::All;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
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

    fn visit_expr_field(&mut self, field: &'tcx hir::ExprField<'tcx>) {
        self.with_lint_attrs(field.hir_id, |builder| {
            intravisit::walk_expr_field(builder, field);
        })
    }

    fn visit_field_def(&mut self, s: &'tcx hir::FieldDef<'tcx>) {
        self.with_lint_attrs(s.hir_id, |builder| {
            intravisit::walk_field_def(builder, s);
        })
    }

    fn visit_variant(&mut self, v: &'tcx hir::Variant<'tcx>) {
        self.with_lint_attrs(v.id, |builder| {
            intravisit::walk_variant(builder, v);
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

    fn visit_pat_field(&mut self, field: &'tcx hir::PatField<'tcx>) {
        self.with_lint_attrs(field.hir_id, |builder| {
            intravisit::walk_pat_field(builder, field);
        })
    }

    fn visit_generic_param(&mut self, p: &'tcx hir::GenericParam<'tcx>) {
        self.with_lint_attrs(p.hir_id, |builder| {
            intravisit::walk_generic_param(builder, p);
        });
    }
}

pub fn provide(providers: &mut Providers) {
    providers.lint_levels = lint_levels;
}
