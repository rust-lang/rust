use crate::consts::{constant_simple, Constant};
use crate::utils::{
    is_expn_of, match_def_path, match_qpath, match_type, method_calls, path_to_res, paths, qpath_res, run_lints,
    snippet, span_lint, span_lint_and_help, span_lint_and_sugg, SpanlessEq,
};
use if_chain::if_chain;
use rustc_ast::ast::{Crate as AstCrate, ItemKind, LitKind, NodeId};
use rustc_ast::visit::FnKind;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::hir_id::CRATE_HIR_ID;
use rustc_hir::intravisit::{NestedVisitorMap, Visitor};
use rustc_hir::{
    BinOpKind, Crate, Expr, ExprKind, HirId, Item, MutTy, Mutability, Node, Path, StmtKind, Ty, TyKind, UnOp,
};
use rustc_lint::{EarlyContext, EarlyLintPass, LateContext, LateLintPass};
use rustc_middle::hir::map::Map;
use rustc_middle::mir::interpret::ConstValue;
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint, impl_lint_pass};
use rustc_span::source_map::{Span, Spanned};
use rustc_span::symbol::{Symbol, SymbolStr};
use rustc_typeck::hir_ty_to_ty;

use std::borrow::{Borrow, Cow};

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

declare_clippy_lint! {
    /// **What it does:** Lints `span_lint_and_then` function calls, where the
    /// closure argument has only one statement and that statement is a method
    /// call to `span_suggestion`, `span_help`, `span_note` (using the same
    /// span), `help` or `note`.
    ///
    /// These usages of `span_lint_and_then` should be replaced with one of the
    /// wrapper functions `span_lint_and_sugg`, span_lint_and_help`, or
    /// `span_lint_and_note`.
    ///
    /// **Why is this bad?** Using the wrapper `span_lint_and_*` functions, is more
    /// convenient, readable and less error prone.
    ///
    /// **Known problems:** None
    ///
    /// *Example:**
    /// Bad:
    /// ```rust,ignore
    /// span_lint_and_then(cx, TEST_LINT, expr.span, lint_msg, |diag| {
    ///     diag.span_suggestion(
    ///         expr.span,
    ///         help_msg,
    ///         sugg.to_string(),
    ///         Applicability::MachineApplicable,
    ///     );
    /// });
    /// span_lint_and_then(cx, TEST_LINT, expr.span, lint_msg, |diag| {
    ///     diag.span_help(expr.span, help_msg);
    /// });
    /// span_lint_and_then(cx, TEST_LINT, expr.span, lint_msg, |diag| {
    ///     diag.help(help_msg);
    /// });
    /// span_lint_and_then(cx, TEST_LINT, expr.span, lint_msg, |diag| {
    ///     diag.span_note(expr.span, note_msg);
    /// });
    /// span_lint_and_then(cx, TEST_LINT, expr.span, lint_msg, |diag| {
    ///     diag.note(note_msg);
    /// });
    /// ```
    ///
    /// Good:
    /// ```rust,ignore
    /// span_lint_and_sugg(
    ///     cx,
    ///     TEST_LINT,
    ///     expr.span,
    ///     lint_msg,
    ///     help_msg,
    ///     sugg.to_string(),
    ///     Applicability::MachineApplicable,
    /// );
    /// span_lint_and_help(cx, TEST_LINT, expr.span, lint_msg, Some(expr.span), help_msg);
    /// span_lint_and_help(cx, TEST_LINT, expr.span, lint_msg, None, help_msg);
    /// span_lint_and_note(cx, TEST_LINT, expr.span, lint_msg, Some(expr.span), note_msg);
    /// span_lint_and_note(cx, TEST_LINT, expr.span, lint_msg, None, note_msg);
    /// ```
    pub COLLAPSIBLE_SPAN_LINT_CALLS,
    internal,
    "found collapsible `span_lint_and_then` calls"
}

declare_clippy_lint! {
    /// **What it does:** Checks for calls to `utils::match_type()` on a type diagnostic item
    /// and suggests to use `utils::is_type_diagnostic_item()` instead.
    ///
    /// **Why is this bad?** `utils::is_type_diagnostic_item()` does not require hardcoded paths.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// Bad:
    /// ```rust,ignore
    /// utils::match_type(cx, ty, &paths::VEC)
    /// ```
    ///
    /// Good:
    /// ```rust,ignore
    /// utils::is_type_diagnostic_item(cx, ty, sym::vec_type)
    /// ```
    pub MATCH_TYPE_ON_DIAGNOSTIC_ITEM,
    internal,
    "using `utils::match_type()` instead of `utils::is_type_diagnostic_item()`"
}

declare_clippy_lint! {
    /// **What it does:**
    /// Checks the paths module for invalid paths.
    ///
    /// **Why is this bad?**
    /// It indicates a bug in the code.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:** None.
    pub INVALID_PATHS,
    internal,
    "invalid path"
}

declare_clippy_lint! {
    /// **What it does:**
    /// Checks for interning symbols that have already been pre-interned and defined as constants.
    ///
    /// **Why is this bad?**
    /// It's faster and easier to use the symbol constant.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// Bad:
    /// ```rust,ignore
    /// let _ = sym!(f32);
    /// ```
    ///
    /// Good:
    /// ```rust,ignore
    /// let _ = sym::f32;
    /// ```
    pub INTERNING_DEFINED_SYMBOL,
    internal,
    "interning a symbol that is pre-interned and defined as a constant"
}

declare_clippy_lint! {
    /// **What it does:** Checks for unnecessary conversion from Symbol to a string.
    ///
    /// **Why is this bad?** It's faster use symbols directly intead of strings.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// Bad:
    /// ```rust,ignore
    /// symbol.as_str() == "clippy";
    /// ```
    ///
    /// Good:
    /// ```rust,ignore
    /// symbol == sym::clippy;
    /// ```
    pub UNNECESSARY_SYMBOL_STR,
    internal,
    "unnecessary conversion between Symbol and string"
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
    declared_lints: FxHashMap<Symbol, Span>,
    registered_lints: FxHashSet<Symbol>,
}

impl_lint_pass!(LintWithoutLintPass => [DEFAULT_LINT, LINT_WITHOUT_LINT_PASS]);

impl<'tcx> LateLintPass<'tcx> for LintWithoutLintPass {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if !run_lints(cx, &[DEFAULT_LINT], item.hir_id) {
            return;
        }

        if let hir::ItemKind::Static(ref ty, Mutability::Not, body_id) = item.kind {
            if is_lint_ref_type(cx, ty) {
                let expr = &cx.tcx.hir().body(body_id).value;
                if_chain! {
                    if let ExprKind::AddrOf(_, _, ref inner_exp) = expr.kind;
                    if let ExprKind::Struct(_, ref fields, _) = inner_exp.kind;
                    let field = fields
                        .iter()
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
            if let hir::ItemKind::Impl(hir::Impl {
                of_trait: None,
                items: ref impl_item_refs,
                ..
            }) = item.kind
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

    fn check_crate_post(&mut self, cx: &LateContext<'tcx>, _: &'tcx Crate<'_>) {
        if !run_lints(cx, &[LINT_WITHOUT_LINT_PASS], CRATE_HIR_ID) {
            return;
        }

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

fn is_lint_ref_type<'tcx>(cx: &LateContext<'tcx>, ty: &Ty<'_>) -> bool {
    if let TyKind::Rptr(
        _,
        MutTy {
            ty: ref inner,
            mutbl: Mutability::Not,
        },
    ) = ty.kind
    {
        if let TyKind::Path(ref path) = inner.kind {
            if let Res::Def(DefKind::Struct, def_id) = cx.qpath_res(path, inner.hir_id) {
                return match_def_path(cx, def_id, &paths::LINT);
            }
        }
    }

    false
}

struct LintCollector<'a, 'tcx> {
    output: &'a mut FxHashSet<Symbol>,
    cx: &'a LateContext<'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for LintCollector<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_path(&mut self, path: &'tcx Path<'_>, _: HirId) {
        if path.segments.len() == 1 {
            self.output.insert(path.segments[0].ident.name);
        }
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::All(self.cx.tcx.hir())
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
        map.insert("span_lint_note", "utils::span_lint_and_note");
        map.insert("span_lint_help", "utils::span_lint_and_help");
        Self { map }
    }
}

impl_lint_pass!(CompilerLintFunctions => [COMPILER_LINT_FUNCTIONS]);

impl<'tcx> LateLintPass<'tcx> for CompilerLintFunctions {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if !run_lints(cx, &[COMPILER_LINT_FUNCTIONS], expr.hir_id) {
            return;
        }

        if_chain! {
            if let ExprKind::MethodCall(ref path, _, ref args, _) = expr.kind;
            let fn_name = path.ident;
            if let Some(sugg) = self.map.get(&*fn_name.as_str());
            let ty = cx.typeck_results().expr_ty(&args[0]).peel_refs();
            if match_type(cx, ty, &paths::EARLY_CONTEXT)
                || match_type(cx, ty, &paths::LATE_CONTEXT);
            then {
                span_lint_and_help(
                    cx,
                    COMPILER_LINT_FUNCTIONS,
                    path.ident.span,
                    "usage of a compiler lint function",
                    None,
                    &format!("please use the Clippy variant of this function: `{}`", sugg),
                );
            }
        }
    }
}

declare_lint_pass!(OuterExpnDataPass => [OUTER_EXPN_EXPN_DATA]);

impl<'tcx> LateLintPass<'tcx> for OuterExpnDataPass {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if !run_lints(cx, &[OUTER_EXPN_EXPN_DATA], expr.hir_id) {
            return;
        }

        let (method_names, arg_lists, spans) = method_calls(expr, 2);
        let method_names: Vec<SymbolStr> = method_names.iter().map(|s| s.as_str()).collect();
        let method_names: Vec<&str> = method_names.iter().map(|s| &**s).collect();
        if_chain! {
            if let ["expn_data", "outer_expn"] = method_names.as_slice();
            let args = arg_lists[1];
            if args.len() == 1;
            let self_arg = &args[0];
            let self_ty = cx.typeck_results().expr_ty(self_arg).peel_refs();
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
    fn check_fn(&mut self, _: &EarlyContext<'_>, fn_kind: FnKind<'_>, _: Span, _: NodeId) {
        if is_trigger_fn(fn_kind) {
            panic!("Would you like some help with that?");
        }
    }
}

fn is_trigger_fn(fn_kind: FnKind<'_>) -> bool {
    match fn_kind {
        FnKind::Fn(_, ident, ..) => ident.name.as_str() == "it_looks_like_you_are_trying_to_kill_clippy",
        FnKind::Closure(..) => false,
    }
}

declare_lint_pass!(CollapsibleCalls => [COLLAPSIBLE_SPAN_LINT_CALLS]);

impl<'tcx> LateLintPass<'tcx> for CollapsibleCalls {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if !run_lints(cx, &[COLLAPSIBLE_SPAN_LINT_CALLS], expr.hir_id) {
            return;
        }

        if_chain! {
            if let ExprKind::Call(ref func, ref and_then_args) = expr.kind;
            if let ExprKind::Path(ref path) = func.kind;
            if match_qpath(path, &["span_lint_and_then"]);
            if and_then_args.len() == 5;
            if let ExprKind::Closure(_, _, body_id, _, _) = &and_then_args[4].kind;
            let body = cx.tcx.hir().body(*body_id);
            if let ExprKind::Block(block, _) = &body.value.kind;
            let stmts = &block.stmts;
            if stmts.len() == 1 && block.expr.is_none();
            if let StmtKind::Semi(only_expr) = &stmts[0].kind;
            if let ExprKind::MethodCall(ref ps, _, ref span_call_args, _) = &only_expr.kind;
            let and_then_snippets = get_and_then_snippets(cx, and_then_args);
            let mut sle = SpanlessEq::new(cx).deny_side_effects();
            then {
                match &*ps.ident.as_str() {
                    "span_suggestion" if sle.eq_expr(&and_then_args[2], &span_call_args[1]) => {
                        suggest_suggestion(cx, expr, &and_then_snippets, &span_suggestion_snippets(cx, span_call_args));
                    },
                    "span_help" if sle.eq_expr(&and_then_args[2], &span_call_args[1]) => {
                        let help_snippet = snippet(cx, span_call_args[2].span, r#""...""#);
                        suggest_help(cx, expr, &and_then_snippets, help_snippet.borrow(), true);
                    },
                    "span_note" if sle.eq_expr(&and_then_args[2], &span_call_args[1]) => {
                        let note_snippet = snippet(cx, span_call_args[2].span, r#""...""#);
                        suggest_note(cx, expr, &and_then_snippets, note_snippet.borrow(), true);
                    },
                    "help" => {
                        let help_snippet = snippet(cx, span_call_args[1].span, r#""...""#);
                        suggest_help(cx, expr, &and_then_snippets, help_snippet.borrow(), false);
                    }
                    "note" => {
                        let note_snippet = snippet(cx, span_call_args[1].span, r#""...""#);
                        suggest_note(cx, expr, &and_then_snippets, note_snippet.borrow(), false);
                    }
                    _  => (),
                }
            }
        }
    }
}

struct AndThenSnippets<'a> {
    cx: Cow<'a, str>,
    lint: Cow<'a, str>,
    span: Cow<'a, str>,
    msg: Cow<'a, str>,
}

fn get_and_then_snippets<'a, 'hir>(cx: &LateContext<'_>, and_then_snippets: &'hir [Expr<'hir>]) -> AndThenSnippets<'a> {
    let cx_snippet = snippet(cx, and_then_snippets[0].span, "cx");
    let lint_snippet = snippet(cx, and_then_snippets[1].span, "..");
    let span_snippet = snippet(cx, and_then_snippets[2].span, "span");
    let msg_snippet = snippet(cx, and_then_snippets[3].span, r#""...""#);

    AndThenSnippets {
        cx: cx_snippet,
        lint: lint_snippet,
        span: span_snippet,
        msg: msg_snippet,
    }
}

struct SpanSuggestionSnippets<'a> {
    help: Cow<'a, str>,
    sugg: Cow<'a, str>,
    applicability: Cow<'a, str>,
}

fn span_suggestion_snippets<'a, 'hir>(
    cx: &LateContext<'_>,
    span_call_args: &'hir [Expr<'hir>],
) -> SpanSuggestionSnippets<'a> {
    let help_snippet = snippet(cx, span_call_args[2].span, r#""...""#);
    let sugg_snippet = snippet(cx, span_call_args[3].span, "..");
    let applicability_snippet = snippet(cx, span_call_args[4].span, "Applicability::MachineApplicable");

    SpanSuggestionSnippets {
        help: help_snippet,
        sugg: sugg_snippet,
        applicability: applicability_snippet,
    }
}

fn suggest_suggestion(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    and_then_snippets: &AndThenSnippets<'_>,
    span_suggestion_snippets: &SpanSuggestionSnippets<'_>,
) {
    span_lint_and_sugg(
        cx,
        COLLAPSIBLE_SPAN_LINT_CALLS,
        expr.span,
        "this call is collapsible",
        "collapse into",
        format!(
            "span_lint_and_sugg({}, {}, {}, {}, {}, {}, {})",
            and_then_snippets.cx,
            and_then_snippets.lint,
            and_then_snippets.span,
            and_then_snippets.msg,
            span_suggestion_snippets.help,
            span_suggestion_snippets.sugg,
            span_suggestion_snippets.applicability
        ),
        Applicability::MachineApplicable,
    );
}

fn suggest_help(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    and_then_snippets: &AndThenSnippets<'_>,
    help: &str,
    with_span: bool,
) {
    let option_span = if with_span {
        format!("Some({})", and_then_snippets.span)
    } else {
        "None".to_string()
    };

    span_lint_and_sugg(
        cx,
        COLLAPSIBLE_SPAN_LINT_CALLS,
        expr.span,
        "this call is collapsible",
        "collapse into",
        format!(
            "span_lint_and_help({}, {}, {}, {}, {}, {})",
            and_then_snippets.cx,
            and_then_snippets.lint,
            and_then_snippets.span,
            and_then_snippets.msg,
            &option_span,
            help
        ),
        Applicability::MachineApplicable,
    );
}

fn suggest_note(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    and_then_snippets: &AndThenSnippets<'_>,
    note: &str,
    with_span: bool,
) {
    let note_span = if with_span {
        format!("Some({})", and_then_snippets.span)
    } else {
        "None".to_string()
    };

    span_lint_and_sugg(
        cx,
        COLLAPSIBLE_SPAN_LINT_CALLS,
        expr.span,
        "this call is collspible",
        "collapse into",
        format!(
            "span_lint_and_note({}, {}, {}, {}, {}, {})",
            and_then_snippets.cx,
            and_then_snippets.lint,
            and_then_snippets.span,
            and_then_snippets.msg,
            note_span,
            note
        ),
        Applicability::MachineApplicable,
    );
}

declare_lint_pass!(MatchTypeOnDiagItem => [MATCH_TYPE_ON_DIAGNOSTIC_ITEM]);

impl<'tcx> LateLintPass<'tcx> for MatchTypeOnDiagItem {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if !run_lints(cx, &[MATCH_TYPE_ON_DIAGNOSTIC_ITEM], expr.hir_id) {
            return;
        }

        if_chain! {
            // Check if this is a call to utils::match_type()
            if let ExprKind::Call(fn_path, [context, ty, ty_path]) = expr.kind;
            if let ExprKind::Path(fn_qpath) = &fn_path.kind;
            if match_qpath(&fn_qpath, &["utils", "match_type"]);
            // Extract the path to the matched type
            if let Some(segments) = path_to_matched_type(cx, ty_path);
            let segments: Vec<&str> = segments.iter().map(|sym| &**sym).collect();
            if let Some(ty_did) = path_to_res(cx, &segments[..]).and_then(|res| res.opt_def_id());
            // Check if the matched type is a diagnostic item
            let diag_items = cx.tcx.diagnostic_items(ty_did.krate);
            if let Some(item_name) = diag_items.iter().find_map(|(k, v)| if *v == ty_did { Some(k) } else { None });
            then {
                let cx_snippet = snippet(cx, context.span, "_");
                let ty_snippet = snippet(cx, ty.span, "_");

                span_lint_and_sugg(
                    cx,
                    MATCH_TYPE_ON_DIAGNOSTIC_ITEM,
                    expr.span,
                    "usage of `utils::match_type()` on a type diagnostic item",
                    "try",
                    format!("utils::is_type_diagnostic_item({}, {}, sym::{})", cx_snippet, ty_snippet, item_name),
                    Applicability::MaybeIncorrect,
                );
            }
        }
    }
}

fn path_to_matched_type(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> Option<Vec<SymbolStr>> {
    use rustc_hir::ItemKind;

    match &expr.kind {
        ExprKind::AddrOf(.., expr) => return path_to_matched_type(cx, expr),
        ExprKind::Path(qpath) => match qpath_res(cx, qpath, expr.hir_id) {
            Res::Local(hir_id) => {
                let parent_id = cx.tcx.hir().get_parent_node(hir_id);
                if let Some(Node::Local(local)) = cx.tcx.hir().find(parent_id) {
                    if let Some(init) = local.init {
                        return path_to_matched_type(cx, init);
                    }
                }
            },
            Res::Def(DefKind::Const | DefKind::Static, def_id) => {
                if let Some(Node::Item(item)) = cx.tcx.hir().get_if_local(def_id) {
                    if let ItemKind::Const(.., body_id) | ItemKind::Static(.., body_id) = item.kind {
                        let body = cx.tcx.hir().body(body_id);
                        return path_to_matched_type(cx, &body.value);
                    }
                }
            },
            _ => {},
        },
        ExprKind::Array(exprs) => {
            let segments: Vec<SymbolStr> = exprs
                .iter()
                .filter_map(|expr| {
                    if let ExprKind::Lit(lit) = &expr.kind {
                        if let LitKind::Str(sym, _) = lit.node {
                            return Some(sym.as_str());
                        }
                    }

                    None
                })
                .collect();

            if segments.len() == exprs.len() {
                return Some(segments);
            }
        },
        _ => {},
    }

    None
}

// This is not a complete resolver for paths. It works on all the paths currently used in the paths
// module.  That's all it does and all it needs to do.
pub fn check_path(cx: &LateContext<'_>, path: &[&str]) -> bool {
    if path_to_res(cx, path).is_some() {
        return true;
    }

    // Some implementations can't be found by `path_to_res`, particularly inherent
    // implementations of native types. Check lang items.
    let path_syms: Vec<_> = path.iter().map(|p| Symbol::intern(p)).collect();
    let lang_items = cx.tcx.lang_items();
    for lang_item in lang_items.items() {
        if let Some(def_id) = lang_item {
            let lang_item_path = cx.get_def_path(*def_id);
            if path_syms.starts_with(&lang_item_path) {
                if let [item] = &path_syms[lang_item_path.len()..] {
                    for child in cx.tcx.item_children(*def_id) {
                        if child.ident.name == *item {
                            return true;
                        }
                    }
                }
            }
        }
    }

    false
}

declare_lint_pass!(InvalidPaths => [INVALID_PATHS]);

impl<'tcx> LateLintPass<'tcx> for InvalidPaths {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        let local_def_id = &cx.tcx.parent_module(item.hir_id);
        let mod_name = &cx.tcx.item_name(local_def_id.to_def_id());
        if_chain! {
            if mod_name.as_str() == "paths";
            if let hir::ItemKind::Const(ty, body_id) = item.kind;
            let ty = hir_ty_to_ty(cx.tcx, ty);
            if let ty::Array(el_ty, _) = &ty.kind();
            if let ty::Ref(_, el_ty, _) = &el_ty.kind();
            if el_ty.is_str();
            let body = cx.tcx.hir().body(body_id);
            let typeck_results = cx.tcx.typeck_body(body_id);
            if let Some(Constant::Vec(path)) = constant_simple(cx, typeck_results, &body.value);
            let path: Vec<&str> = path.iter().map(|x| {
                    if let Constant::Str(s) = x {
                        s.as_str()
                    } else {
                        // We checked the type of the constant above
                        unreachable!()
                    }
                }).collect();
            if !check_path(cx, &path[..]);
            then {
                span_lint(cx, CLIPPY_LINTS_INTERNAL, item.span, "invalid path");
            }
        }
    }
}

#[derive(Default)]
pub struct InterningDefinedSymbol {
    // Maps the symbol value to the constant DefId.
    symbol_map: FxHashMap<u32, DefId>,
}

impl_lint_pass!(InterningDefinedSymbol => [INTERNING_DEFINED_SYMBOL, UNNECESSARY_SYMBOL_STR]);

impl<'tcx> LateLintPass<'tcx> for InterningDefinedSymbol {
    fn check_crate(&mut self, cx: &LateContext<'_>, _: &Crate<'_>) {
        if !self.symbol_map.is_empty() {
            return;
        }

        for &module in &[&paths::KW_MODULE, &paths::SYM_MODULE] {
            if let Some(Res::Def(_, def_id)) = path_to_res(cx, module) {
                for item in cx.tcx.item_children(def_id).iter() {
                    if_chain! {
                        if let Res::Def(DefKind::Const, item_def_id) = item.res;
                        let ty = cx.tcx.type_of(item_def_id);
                        if match_type(cx, ty, &paths::SYMBOL);
                        if let Ok(ConstValue::Scalar(value)) = cx.tcx.const_eval_poly(item_def_id);
                        if let Ok(value) = value.to_u32();
                        then {
                            self.symbol_map.insert(value, item_def_id);
                        }
                    }
                }
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if let ExprKind::Call(func, [arg]) = &expr.kind;
            if let ty::FnDef(def_id, _) = cx.typeck_results().expr_ty(func).kind();
            if match_def_path(cx, *def_id, &paths::SYMBOL_INTERN);
            if let Some(Constant::Str(arg)) = constant_simple(cx, cx.typeck_results(), arg);
            let value = Symbol::intern(&arg).as_u32();
            if let Some(&def_id) = self.symbol_map.get(&value);
            then {
                span_lint_and_sugg(
                    cx,
                    INTERNING_DEFINED_SYMBOL,
                    is_expn_of(expr.span, "sym").unwrap_or(expr.span),
                    "interning a defined symbol",
                    "try",
                    cx.tcx.def_path_str(def_id),
                    Applicability::MachineApplicable,
                );
            }
        }
        if let ExprKind::Binary(op, left, right) = expr.kind {
            if matches!(op.node, BinOpKind::Eq | BinOpKind::Ne) {
                let data = [
                    (left, self.symbol_str_expr(left, cx)),
                    (right, self.symbol_str_expr(right, cx)),
                ];
                match data {
                    // both operands are a symbol string
                    [(_, Some(left)), (_, Some(right))] => {
                        span_lint_and_sugg(
                            cx,
                            UNNECESSARY_SYMBOL_STR,
                            expr.span,
                            "unnecessary `Symbol` to string conversion",
                            "try",
                            format!(
                                "{} {} {}",
                                left.as_symbol_snippet(cx),
                                op.node.as_str(),
                                right.as_symbol_snippet(cx),
                            ),
                            Applicability::MachineApplicable,
                        );
                    },
                    // one of the operands is a symbol string
                    [(expr, Some(symbol)), _] | [_, (expr, Some(symbol))] => {
                        // creating an owned string for comparison
                        if matches!(symbol, SymbolStrExpr::Expr { is_to_owned: true, .. }) {
                            span_lint_and_sugg(
                                cx,
                                UNNECESSARY_SYMBOL_STR,
                                expr.span,
                                "unnecessary string allocation",
                                "try",
                                format!("{}.as_str()", symbol.as_symbol_snippet(cx)),
                                Applicability::MachineApplicable,
                            );
                        }
                    },
                    // nothing found
                    [(_, None), (_, None)] => {},
                }
            }
        }
    }
}

impl InterningDefinedSymbol {
    fn symbol_str_expr<'tcx>(&self, expr: &'tcx Expr<'tcx>, cx: &LateContext<'tcx>) -> Option<SymbolStrExpr<'tcx>> {
        static IDENT_STR_PATHS: &[&[&str]] = &[&paths::IDENT_AS_STR, &paths::TO_STRING_METHOD];
        static SYMBOL_STR_PATHS: &[&[&str]] = &[
            &paths::SYMBOL_AS_STR,
            &paths::SYMBOL_TO_IDENT_STRING,
            &paths::TO_STRING_METHOD,
        ];
        // SymbolStr might be de-referenced: `&*symbol.as_str()`
        let call = if_chain! {
            if let ExprKind::AddrOf(_, _, e) = expr.kind;
            if let ExprKind::Unary(UnOp::UnDeref, e) = e.kind;
            then { e } else { expr }
        };
        if_chain! {
            // is a method call
            if let ExprKind::MethodCall(_, _, [item], _) = call.kind;
            if let Some(did) = cx.typeck_results().type_dependent_def_id(call.hir_id);
            let ty = cx.typeck_results().expr_ty(item);
            // ...on either an Ident or a Symbol
            if let Some(is_ident) = if match_type(cx, ty, &paths::SYMBOL) {
                Some(false)
            } else if match_type(cx, ty, &paths::IDENT) {
                Some(true)
            } else {
                None
            };
            // ...which converts it to a string
            let paths = if is_ident { IDENT_STR_PATHS } else { SYMBOL_STR_PATHS };
            if let Some(path) = paths.iter().find(|path| match_def_path(cx, did, path));
            then {
                let is_to_owned = path.last().unwrap().ends_with("string");
                return Some(SymbolStrExpr::Expr {
                    item,
                    is_ident,
                    is_to_owned,
                });
            }
        }
        // is a string constant
        if let Some(Constant::Str(s)) = constant_simple(cx, cx.typeck_results(), expr) {
            let value = Symbol::intern(&s).as_u32();
            // ...which matches a symbol constant
            if let Some(&def_id) = self.symbol_map.get(&value) {
                return Some(SymbolStrExpr::Const(def_id));
            }
        }
        None
    }
}

enum SymbolStrExpr<'tcx> {
    /// a string constant with a corresponding symbol constant
    Const(DefId),
    /// a "symbol to string" expression like `symbol.as_str()`
    Expr {
        /// part that evaluates to `Symbol` or `Ident`
        item: &'tcx Expr<'tcx>,
        is_ident: bool,
        /// whether an owned `String` is created like `to_ident_string()`
        is_to_owned: bool,
    },
}

impl<'tcx> SymbolStrExpr<'tcx> {
    /// Returns a snippet that evaluates to a `Symbol` and is const if possible
    fn as_symbol_snippet(&self, cx: &LateContext<'_>) -> Cow<'tcx, str> {
        match *self {
            Self::Const(def_id) => cx.tcx.def_path_str(def_id).into(),
            Self::Expr { item, is_ident, .. } => {
                let mut snip = snippet(cx, item.span.source_callsite(), "..");
                if is_ident {
                    // get `Ident.name`
                    snip.to_mut().push_str(".name");
                }
                snip
            },
        }
    }
}
