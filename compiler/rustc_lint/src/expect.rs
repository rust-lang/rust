use crate::builtin;
use crate::context::{CheckLintNameResult, LintStore};
use crate::late::unerased_lint_store;
use crate::levels::{try_parse_reason_metadata, ParseLintReasonResult};
use rustc_ast as ast;
use rustc_ast::unwrap_or;
use rustc_ast_pretty::pprust;
use rustc_hir as hir;
use rustc_hir::intravisit;
use rustc_middle::hir::map::Map;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::lint::{Level, LintId};
use rustc_session::Session;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::{MultiSpan, Span};

fn check_expect_lint(tcx: TyCtxt<'_>, _: ()) -> () {
    if !tcx.sess.features_untracked().enabled(sym::lint_reasons) {
        return;
    }

    let store = unerased_lint_store(tcx);
    let krate = tcx.hir().krate();

    let mut checker = LintExpectationChecker::new(tcx, tcx.sess, store);
    checker.check_crate(krate);
}

/// This is used by the expectation check to define in which scope expectations
/// count towards fulfilling the expectation.
#[derive(Debug, Clone, Copy)]
enum CheckScope {
    /// The scope it limited to a `Span` only lint emissions within this span
    /// can fulfill the expectation.
    Span(Span),
    /// All emissions in this crate can fulfill this emission. This is used for
    /// crate expectation attributes.
    CreateWide,
}

impl CheckScope {
    fn includes_span(&self, emission: &MultiSpan) -> bool {
        match self {
            CheckScope::Span(scope_span) => {
                emission.primary_spans().iter().any(|span| scope_span.contains(*span))
            }
            CheckScope::CreateWide => true,
        }
    }
}

#[derive(Debug, Clone)]
struct LintIdEmission {
    lint_id: LintId,
    span: MultiSpan,
}

impl LintIdEmission {
    fn new(lint_id: LintId, span: MultiSpan) -> Self {
        Self { lint_id, span }
    }
}

#[derive(Debug, Clone)]
struct LintExpectation {
    lints: Vec<LintId>,
    reason: Option<Symbol>,
    attr_span: Span,
}

impl LintExpectation {
    fn new(lints: Vec<LintId>, reason: Option<Symbol>, attr_span: Span) -> Self {
        Self { lints, reason, attr_span }
    }
}

struct LintExpectationChecker<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    sess: &'a Session,
    store: &'a LintStore,
    emitted_lints: Vec<LintIdEmission>,
    crate_attrs: &'tcx [ast::Attribute],
}

impl<'a, 'tcx> LintExpectationChecker<'a, 'tcx> {
    fn new(tcx: TyCtxt<'tcx>, sess: &'a Session, store: &'a LintStore) -> Self {
        let mut expect_lint_emissions = tcx.sess.diagnostic().steal_expect_lint_emissions();
        let crate_attrs = tcx.hir().attrs(hir::CRATE_HIR_ID);
        let emitted_lints = expect_lint_emissions
            .drain(..)
            .filter_map(|emission| {
                if let CheckLintNameResult::Ok(&[id]) =
                    store.check_lint_name(sess, &emission.lint_name, None, crate_attrs)
                {
                    Some(LintIdEmission::new(id, emission.lint_span))
                } else {
                    None
                }
            })
            .collect();

        Self { tcx, sess, store, emitted_lints, crate_attrs }
    }

    fn check_item_with_attrs<F>(&mut self, id: hir::HirId, scope: Span, f: F)
    where
        F: FnOnce(&mut Self),
    {
        let mut expectations = self.collect_expectations(id);

        f(self);

        for expect in expectations.drain(..) {
            self.check_expectation(expect, CheckScope::Span(scope), id);
        }
    }

    fn check_crate(&mut self, krate: &'tcx hir::Crate<'tcx>) {
        let mut expectations = self.collect_expectations(hir::CRATE_HIR_ID);

        intravisit::walk_crate(self, krate);

        for expect in expectations.drain(..) {
            self.check_expectation(expect, CheckScope::CreateWide, hir::CRATE_HIR_ID);
        }
    }

    fn collect_expectations(&self, id: hir::HirId) -> Vec<LintExpectation> {
        let mut result = Vec::new();

        for attr in self.tcx.hir().attrs(id) {
            // We only care about expectations
            if attr.name_or_empty() != sym::expect {
                continue;
            }

            self.sess.mark_attr_used(attr);

            let mut metas = unwrap_or!(attr.meta_item_list(), continue);
            if metas.is_empty() {
                // FIXME (#55112): issue unused-attributes lint for `#[level()]`
                continue;
            }

            // Before processing the lint names, look for a reason (RFC 2383)
            // at the end.
            let tail_li = &metas[metas.len() - 1];
            let reason = match try_parse_reason_metadata(tail_li, self.sess) {
                ParseLintReasonResult::Ok(reason) => {
                    metas.pop().unwrap();
                    Some(reason)
                }
                ParseLintReasonResult::MalformedReason => {
                    metas.pop().unwrap();
                    None
                }
                ParseLintReasonResult::NotFound => None,
            };

            // This will simply collect the lints specified in the expect attribute.
            // Error handling about unknown renamed and weird lints is done by the
            // `LintLevelMapBuilder`
            let mut lints: Vec<LintId> = Default::default();
            for li in metas {
                let mut meta_item = match li {
                    ast::NestedMetaItem::MetaItem(meta_item) if meta_item.is_word() => meta_item,
                    _ => continue,
                };

                // Extracting the tool
                let tool_name = if meta_item.path.segments.len() > 1 {
                    Some(meta_item.path.segments.remove(0).ident.name)
                } else {
                    None
                };

                // Checking the lint name
                let name = pprust::path_to_string(&meta_item.path);
                match &self.store.check_lint_name(self.sess, &name, tool_name, self.crate_attrs) {
                    CheckLintNameResult::Ok(ids) => {
                        lints.extend_from_slice(ids);
                    }
                    CheckLintNameResult::Tool(result) => {
                        match *result {
                            Ok(ids) => {
                                lints.extend_from_slice(ids);
                            }
                            Err((_, _)) => {
                                // The lint could not be found, this can happen if the
                                // lint doesn't exist in the tool or if the Tool is not
                                // enabled. In either case we don't want to add it to the
                                // lints as it can not be emitted during this compiler run
                                // and the expectation could therefor also not be fulfilled.
                                continue;
                            }
                        }
                    }
                    CheckLintNameResult::Warning(_, Some(new_name)) => {
                        // The lint has been renamed. The `LintLevelMapBuilder` then
                        // registers the level for the new name. This means that the
                        // expectation of a renamed lint should also be fulfilled by
                        // the new name of the lint.

                        // NOTE: `new_name` already includes the tool name, so we don't have to add it again.
                        if let CheckLintNameResult::Ok(ids) =
                            self.store.check_lint_name(self.sess, &new_name, None, self.crate_attrs)
                        {
                            lints.extend_from_slice(ids);
                        }
                    }
                    CheckLintNameResult::Warning(_, _)
                    | CheckLintNameResult::NoLint(_)
                    | CheckLintNameResult::NoTool => {
                        // The `LintLevelMapBuilder` will issue a message about this.
                        continue;
                    }
                }
            }

            if !lints.is_empty() {
                result.push(LintExpectation::new(lints, reason, attr.span));
            }
        }

        result
    }

    fn check_expectation(
        &mut self,
        expectation: LintExpectation,
        scope: CheckScope,
        id: hir::HirId,
    ) {
        let mut fulfilled = false;
        let mut index = 0;
        while index < self.emitted_lints.len() {
            let lint_emission = &self.emitted_lints[index];
            let lint = &lint_emission.lint_id;

            if expectation.lints.contains(lint) && scope.includes_span(&lint_emission.span) {
                drop(self.emitted_lints.swap_remove(index));
                fulfilled = true;

                // The index is not increase here as the entry in the
                // index has been changed.
                continue;
            }
            index += 1;
        }

        if !fulfilled {
            self.emit_unfulfilled_expectation_lint(&expectation, expectation.attr_span, id);
        }
    }

    fn emit_unfulfilled_expectation_lint(
        &mut self,
        expectation: &LintExpectation,
        span: Span,
        id: hir::HirId,
    ) {
        let parent_id = self.tcx.hir().get_parent_node(id);
        let level = self.tcx.lint_level_at_node(builtin::UNFULFILLED_LINT_EXPECTATION, parent_id).0;
        if level == Level::Expect {
            // This diagnostic is actually expected. It has to be added manually to
            // `self.emitted_lints` because we only collect expected diagnostics at
            // the start. It would therefore not be included in the backlog.
            let expect_lint_name = builtin::UNFULFILLED_LINT_EXPECTATION.name.to_ascii_lowercase();
            if let CheckLintNameResult::Ok(&[expect_lint_id]) =
                self.store.check_lint_name(self.sess, &expect_lint_name, None, self.crate_attrs)
            {
                self.emitted_lints.push(LintIdEmission::new(expect_lint_id, span.into()));
            } else {
                unreachable!(
                    "the `unfulfilled_lint_expectation` lint should be registered when this code is executed"
                );
            }

            // The diagnostic will still be emitted as usual to make sure that it's
            // stored in cache.
        }

        self.tcx.struct_span_lint_hir(
            builtin::UNFULFILLED_LINT_EXPECTATION,
            parent_id,
            span,
            |diag| {
                let mut diag = diag.build("this lint expectation is unfulfilled");
                if let Some(rationale) = expectation.reason {
                    diag.note(&rationale.as_str());
                }
                diag.emit();
            },
        );
    }
}

impl<'tcx> intravisit::Visitor<'tcx> for LintExpectationChecker<'_, 'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
        intravisit::NestedVisitorMap::All(self.tcx.hir())
    }

    fn visit_param(&mut self, param: &'tcx hir::Param<'tcx>) {
        self.check_item_with_attrs(param.hir_id, param.span, |builder| {
            intravisit::walk_param(builder, param);
        });
    }

    fn visit_item(&mut self, it: &'tcx hir::Item<'tcx>) {
        self.check_item_with_attrs(it.hir_id(), it.span, |builder| {
            intravisit::walk_item(builder, it);
        });
    }

    fn visit_foreign_item(&mut self, it: &'tcx hir::ForeignItem<'tcx>) {
        self.check_item_with_attrs(it.hir_id(), it.span, |builder| {
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
        self.check_item_with_attrs(e.hir_id, e.span, |builder| {
            intravisit::walk_expr(builder, e);
        })
    }

    fn visit_field_def(&mut self, s: &'tcx hir::FieldDef<'tcx>) {
        self.check_item_with_attrs(s.hir_id, s.span, |builder| {
            intravisit::walk_field_def(builder, s);
        })
    }

    fn visit_variant(
        &mut self,
        v: &'tcx hir::Variant<'tcx>,
        g: &'tcx hir::Generics<'tcx>,
        item_id: hir::HirId,
    ) {
        self.check_item_with_attrs(v.id, v.span, |builder| {
            intravisit::walk_variant(builder, v, g, item_id);
        })
    }

    fn visit_local(&mut self, l: &'tcx hir::Local<'tcx>) {
        self.check_item_with_attrs(l.hir_id, l.span, |builder| {
            intravisit::walk_local(builder, l);
        })
    }

    fn visit_arm(&mut self, a: &'tcx hir::Arm<'tcx>) {
        self.check_item_with_attrs(a.hir_id, a.span, |builder| {
            intravisit::walk_arm(builder, a);
        })
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem<'tcx>) {
        self.check_item_with_attrs(trait_item.hir_id(), trait_item.span, |builder| {
            intravisit::walk_trait_item(builder, trait_item);
        });
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem<'tcx>) {
        self.check_item_with_attrs(impl_item.hir_id(), impl_item.span, |builder| {
            intravisit::walk_impl_item(builder, impl_item);
        });
    }
}

pub fn provide(providers: &mut Providers) {
    providers.check_expect_lint = check_expect_lint;
}
