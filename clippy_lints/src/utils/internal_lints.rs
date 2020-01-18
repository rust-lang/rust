use crate::utils::{
    is_expn_of, match_def_path, match_type, method_calls, paths, span_help_and_lint, span_lint, span_lint_and_sugg,
    walk_ptrs_ty,
};
use if_chain::if_chain;
use rustc::hir::map::Map;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use rustc_hir::*;
use rustc_lint::{EarlyContext, EarlyLintPass, LateContext, LateLintPass};
use rustc_session::declare_tool_lint;
use rustc_session::{declare_lint_pass, impl_lint_pass};
use rustc_span::source_map::{Span, Spanned};
use rustc_span::symbol::SymbolStr;
use syntax::ast;
use syntax::ast::{Crate as AstCrate, ItemKind, LitKind, Name};
use syntax::visit::FnKind;

declare_clippy_lint! {
    /// **What it does:** Checks for various things we like to keep tidy in clippy.
    ///
    /// **Why is this bad?** We like to pretend we're an example of tidy code.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:** Wrong ordering of the util::paths constants.
    pub CLIPPY_LINTS_INTERNAL,
    internal,
    "various things that will negatively affect your clippy experience"
}

declare_clippy_lint! {
    /// **What it does:** Ensures every lint is associated to a `LintPass`.
    ///
    /// **Why is this bad?** The compiler only knows lints via a `LintPass`. Without
    /// putting a lint to a `LintPass::get_lints()`'s return, the compiler will not
    /// know the name of the lint.
    ///
    /// **Known problems:** Only checks for lints associated using the
    /// `declare_lint_pass!`, `impl_lint_pass!`, and `lint_array!` macros.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// declare_lint! { pub LINT_1, ... }
    /// declare_lint! { pub LINT_2, ... }
    /// declare_lint! { pub FORGOTTEN_LINT, ... }
    /// // ...
    /// declare_lint_pass!(Pass => [LINT_1, LINT_2]);
    /// // missing FORGOTTEN_LINT
    /// ```
    pub LINT_WITHOUT_LINT_PASS,
    internal,
    "declaring a lint without associating it in a LintPass"
}

declare_clippy_lint! {
    /// **What it does:** Checks for calls to `cx.span_lint*` and suggests to use the `utils::*`
    /// variant of the function.
    ///
    /// **Why is this bad?** The `utils::*` variants also add a link to the Clippy documentation to the
    /// warning/error messages.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// Bad:
    /// ```rust,ignore
    /// cx.span_lint(LINT_NAME, "message");
    /// ```
    ///
    /// Good:
    /// ```rust,ignore
    /// utils::span_lint(cx, LINT_NAME, "message");
    /// ```
    pub COMPILER_LINT_FUNCTIONS,
    internal,
    "usage of the lint functions of the compiler instead of the utils::* variant"
}

declare_clippy_lint! {
    /// **What it does:** Checks for calls to `cx.outer().expn_data()` and suggests to use
    /// the `cx.outer_expn_data()`
    ///
    /// **Why is this bad?** `cx.outer_expn_data()` is faster and more concise.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// Bad:
    /// ```rust,ignore
    /// expr.span.ctxt().outer().expn_data()
    /// ```
    ///
    /// Good:
    /// ```rust,ignore
    /// expr.span.ctxt().outer_expn_data()
    /// ```
    pub OUTER_EXPN_EXPN_DATA,
    internal,
    "using `cx.outer_expn().expn_data()` instead of `cx.outer_expn_data()`"
}

declare_clippy_lint! {
    /// **What it does:** Not an actual lint. This lint is only meant for testing our customized internal compiler
    /// error message by calling `panic`.
    ///
    /// **Why is this bad?** ICE in large quantities can damage your teeth
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    /// Bad:
    /// ```rust,ignore
    /// 🍦🍦🍦🍦🍦
    /// ```
    pub PRODUCE_ICE,
    internal,
    "this message should not appear anywhere as we ICE before and don't emit the lint"
}

declare_clippy_lint! {
    /// **What it does:** Checks for cases of an auto-generated lint without an updated description,
    /// i.e. `default lint description`.
    ///
    /// **Why is this bad?** Indicates that the lint is not finished.
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    /// Bad:
    /// ```rust,ignore
    /// declare_lint! { pub COOL_LINT, nursery, "default lint description" }
    /// ```
    ///
    /// Good:
    /// ```rust,ignore
    /// declare_lint! { pub COOL_LINT, nursery, "a great new lint" }
    /// ```
    pub DEFAULT_LINT,
    internal,
    "found 'default lint description' in a lint declaration"
}

declare_lint_pass!(ClippyLintsInternal => [CLIPPY_LINTS_INTERNAL]);

impl EarlyLintPass for ClippyLintsInternal {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, krate: &AstCrate) {
        if let Some(utils) = krate
            .module
            .items
            .iter()
            .find(|item| item.ident.name.as_str() == "utils")
        {
            if let ItemKind::Mod(ref utils_mod) = utils.kind {
                if let Some(paths) = utils_mod.items.iter().find(|item| item.ident.name.as_str() == "paths") {
                    if let ItemKind::Mod(ref paths_mod) = paths.kind {
                        let mut last_name: Option<SymbolStr> = None;
                        for item in &*paths_mod.items {
                            let name = item.ident.as_str();
                            if let Some(ref last_name) = last_name {
                                if **last_name > *name {
                                    span_lint(
                                        cx,
                                        CLIPPY_LINTS_INTERNAL,
                                        item.span,
                                        "this constant should be before the previous constant due to lexical \
                                         ordering",
                                    );
                                }
                            }
                            last_name = Some(name);
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LintWithoutLintPass {
    declared_lints: FxHashMap<Name, Span>,
    registered_lints: FxHashSet<Name>,
}

impl_lint_pass!(LintWithoutLintPass => [DEFAULT_LINT, LINT_WITHOUT_LINT_PASS]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for LintWithoutLintPass {
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item<'_>) {
        if let hir::ItemKind::Static(ref ty, Mutability::Not, body_id) = item.kind {
            if is_lint_ref_type(cx, ty) {
                let expr = &cx.tcx.hir().body(body_id).value;
                if_chain! {
                    if let ExprKind::AddrOf(_, _, ref inner_exp) = expr.kind;
                    if let ExprKind::Struct(_, ref fields, _) = inner_exp.kind;
                    let field = fields.iter()
                                      .find(|f| f.ident.as_str() == "desc")
                                      .expect("lints must have a description field");
                    if let ExprKind::Lit(Spanned {
                        node: LitKind::Str(ref sym, _),
                        ..
                    }) = field.expr.kind;
                    if sym.as_str() == "default lint description";

                    then {
                        span_lint(
                            cx,
                            DEFAULT_LINT,
                            item.span,
                            &format!("the lint `{}` has the default lint description", item.ident.name),
                        );
                    }
                }
                self.declared_lints.insert(item.ident.name, item.span);
            }
        } else if is_expn_of(item.span, "impl_lint_pass").is_some()
            || is_expn_of(item.span, "declare_lint_pass").is_some()
        {
            if let hir::ItemKind::Impl {
                of_trait: None,
                items: ref impl_item_refs,
                ..
            } = item.kind
            {
                let mut collector = LintCollector {
                    output: &mut self.registered_lints,
                    cx,
                };
                let body_id = cx.tcx.hir().body_owned_by(
                    impl_item_refs
                        .iter()
                        .find(|iiref| iiref.ident.as_str() == "get_lints")
                        .expect("LintPass needs to implement get_lints")
                        .id
                        .hir_id,
                );
                collector.visit_expr(&cx.tcx.hir().body(body_id).value);
            }
        }
    }

    fn check_crate_post(&mut self, cx: &LateContext<'a, 'tcx>, _: &'tcx Crate<'_>) {
        for (lint_name, &lint_span) in &self.declared_lints {
            // When using the `declare_tool_lint!` macro, the original `lint_span`'s
            // file points to "<rustc macros>".
            // `compiletest-rs` thinks that's an error in a different file and
            // just ignores it. This causes the test in compile-fail/lint_pass
            // not able to capture the error.
            // Therefore, we need to climb the macro expansion tree and find the
            // actual span that invoked `declare_tool_lint!`:
            let lint_span = lint_span.ctxt().outer_expn_data().call_site;

            if !self.registered_lints.contains(lint_name) {
                span_lint(
                    cx,
                    LINT_WITHOUT_LINT_PASS,
                    lint_span,
                    &format!("the lint `{}` is not added to any `LintPass`", lint_name),
                );
            }
        }
    }
}

fn is_lint_ref_type<'tcx>(cx: &LateContext<'_, 'tcx>, ty: &Ty<'_>) -> bool {
    if let TyKind::Rptr(
        _,
        MutTy {
            ty: ref inner,
            mutbl: Mutability::Not,
        },
    ) = ty.kind
    {
        if let TyKind::Path(ref path) = inner.kind {
            if let Res::Def(DefKind::Struct, def_id) = cx.tables.qpath_res(path, inner.hir_id) {
                return match_def_path(cx, def_id, &paths::LINT);
            }
        }
    }

    false
}

struct LintCollector<'a, 'tcx> {
    output: &'a mut FxHashSet<Name>,
    cx: &'a LateContext<'a, 'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for LintCollector<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        walk_expr(self, expr);
    }

    fn visit_path(&mut self, path: &'tcx Path<'_>, _: HirId) {
        if path.segments.len() == 1 {
            self.output.insert(path.segments[0].ident.name);
        }
    }
    fn nested_visit_map(&mut self) -> NestedVisitorMap<'_, Self::Map> {
        NestedVisitorMap::All(&self.cx.tcx.hir())
    }
}

#[derive(Clone, Default)]
pub struct CompilerLintFunctions {
    map: FxHashMap<&'static str, &'static str>,
}

impl CompilerLintFunctions {
    #[must_use]
    pub fn new() -> Self {
        let mut map = FxHashMap::default();
        map.insert("span_lint", "utils::span_lint");
        map.insert("struct_span_lint", "utils::span_lint");
        map.insert("lint", "utils::span_lint");
        map.insert("span_lint_note", "utils::span_note_and_lint");
        map.insert("span_lint_help", "utils::span_help_and_lint");
        Self { map }
    }
}

impl_lint_pass!(CompilerLintFunctions => [COMPILER_LINT_FUNCTIONS]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for CompilerLintFunctions {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if let ExprKind::MethodCall(ref path, _, ref args) = expr.kind;
            let fn_name = path.ident;
            if let Some(sugg) = self.map.get(&*fn_name.as_str());
            let ty = walk_ptrs_ty(cx.tables.expr_ty(&args[0]));
            if match_type(cx, ty, &paths::EARLY_CONTEXT)
                || match_type(cx, ty, &paths::LATE_CONTEXT);
            then {
                span_help_and_lint(
                    cx,
                    COMPILER_LINT_FUNCTIONS,
                    path.ident.span,
                    "usage of a compiler lint function",
                    &format!("please use the Clippy variant of this function: `{}`", sugg),
                );
            }
        }
    }
}

declare_lint_pass!(OuterExpnDataPass => [OUTER_EXPN_EXPN_DATA]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for OuterExpnDataPass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx hir::Expr<'_>) {
        let (method_names, arg_lists, spans) = method_calls(expr, 2);
        let method_names: Vec<SymbolStr> = method_names.iter().map(|s| s.as_str()).collect();
        let method_names: Vec<&str> = method_names.iter().map(|s| &**s).collect();
        if_chain! {
            if let ["expn_data", "outer_expn"] = method_names.as_slice();
            let args = arg_lists[1];
            if args.len() == 1;
            let self_arg = &args[0];
            let self_ty = walk_ptrs_ty(cx.tables.expr_ty(self_arg));
            if match_type(cx, self_ty, &paths::SYNTAX_CONTEXT);
            then {
                span_lint_and_sugg(
                    cx,
                    OUTER_EXPN_EXPN_DATA,
                    spans[1].with_hi(expr.span.hi()),
                    "usage of `outer_expn().expn_data()`",
                    "try",
                    "outer_expn_data()".to_string(),
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}

declare_lint_pass!(ProduceIce => [PRODUCE_ICE]);

impl EarlyLintPass for ProduceIce {
    fn check_fn(&mut self, _: &EarlyContext<'_>, fn_kind: FnKind<'_>, _: &ast::FnDecl, _: Span, _: ast::NodeId) {
        if is_trigger_fn(fn_kind) {
            panic!("Testing the ICE message");
        }
    }
}

fn is_trigger_fn(fn_kind: FnKind<'_>) -> bool {
    match fn_kind {
        FnKind::ItemFn(ident, ..) | FnKind::Method(ident, ..) => {
            ident.name.as_str() == "it_looks_like_you_are_trying_to_kill_clippy"
        },
        FnKind::Closure(..) => false,
    }
}
