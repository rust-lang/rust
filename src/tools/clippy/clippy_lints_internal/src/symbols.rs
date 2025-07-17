use crate::internal_paths;
use clippy_utils::diagnostics::span_lint_and_then;
use rustc_ast::LitKind;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Expr, ExprKind, Lit, Node, Pat, PatExprKind, PatKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::mir::ConstValue;
use rustc_middle::ty;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::symbol::Symbol;
use rustc_span::{Span, sym};

declare_tool_lint! {
    /// ### What it does
    /// Checks for interning string literals as symbols
    ///
    /// ### Why is this bad?
    /// It's faster and easier to use the symbol constant. If one doesn't exist it can be added to `clippy_utils/src/sym.rs`
    ///
    /// ### Example
    /// ```rust,ignore
    /// let _ = Symbol::intern("f32");
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// let _ = sym::f32;
    /// ```
    pub clippy::INTERNING_LITERALS,
    Warn,
    "interning a symbol that is a literal",
    report_in_external_macro: true
}

declare_tool_lint! {
    /// ### What it does
    /// Checks for calls to `Symbol::as_str`
    ///
    /// ### Why is this bad?
    /// It's faster and easier to use the symbol constant. If one doesn't exist it can be added to `clippy_utils/src/sym.rs`
    ///
    /// ### Example
    /// ```rust,ignore
    /// symbol.as_str() == "foo"
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// symbol == sym::foo
    /// ```
    pub clippy::SYMBOL_AS_STR,
    Warn,
    "calls to `Symbol::as_str`",
    report_in_external_macro: true
}

#[derive(Default)]
pub struct Symbols {
    // Maps the symbol to the import path
    symbol_map: FxHashMap<u32, (&'static str, Symbol)>,
}

impl_lint_pass!(Symbols => [INTERNING_LITERALS, SYMBOL_AS_STR]);

impl Symbols {
    fn lit_suggestion(&self, lit: Lit) -> Option<(Span, String)> {
        if let LitKind::Str(name, _) = lit.node {
            let sugg = if let Some((prefix, name)) = self.symbol_map.get(&name.as_u32()) {
                format!("{prefix}::{name}")
            } else {
                format!("sym::{}", name.as_str().replace(|ch: char| !ch.is_alphanumeric(), "_"))
            };
            Some((lit.span, sugg))
        } else {
            None
        }
    }

    fn expr_suggestion(&self, expr: &Expr<'_>) -> Option<(Span, String)> {
        if let ExprKind::Lit(lit) = expr.kind {
            self.lit_suggestion(lit)
        } else {
            None
        }
    }

    fn pat_suggestions(&self, pat: &Pat<'_>, suggestions: &mut Vec<(Span, String)>) {
        pat.walk_always(|pat| {
            if let PatKind::Expr(pat_expr) = pat.kind
                && let PatExprKind::Lit { lit, .. } = pat_expr.kind
            {
                suggestions.extend(self.lit_suggestion(lit));
            }
        });
    }
}

impl<'tcx> LateLintPass<'tcx> for Symbols {
    fn check_crate(&mut self, cx: &LateContext<'_>) {
        let modules = [
            ("kw", &internal_paths::KW_MODULE),
            ("sym", &internal_paths::SYM_MODULE),
            ("sym", &internal_paths::CLIPPY_SYM_MODULE),
        ];
        for (prefix, module) in modules {
            for def_id in module.get(cx) {
                // When linting `clippy_utils` itself we can't use `module_children` as it's a local def id. It will
                // still lint but the suggestion may suggest the incorrect name for symbols such as `sym::CRLF`
                if def_id.is_local() {
                    continue;
                }

                for item in cx.tcx.module_children(def_id) {
                    if let Res::Def(DefKind::Const, item_def_id) = item.res
                        && let ty = cx.tcx.type_of(item_def_id).instantiate_identity()
                        && internal_paths::SYMBOL.matches_ty(cx, ty)
                        && let Ok(ConstValue::Scalar(value)) = cx.tcx.const_eval_poly(item_def_id)
                        && let Some(value) = value.to_u32().discard_err()
                    {
                        self.symbol_map.insert(value, (prefix, item.ident.name));
                    }
                }
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Call(func, [arg]) = &expr.kind
            && let ty::FnDef(def_id, _) = cx.typeck_results().expr_ty(func).kind()
            && cx.tcx.is_diagnostic_item(sym::SymbolIntern, *def_id)
            && let Some((_, sugg)) = self.expr_suggestion(arg)
        {
            span_lint_and_then(
                cx,
                INTERNING_LITERALS,
                expr.span,
                "interning a string literal",
                |diag| {
                    diag.span_suggestion_verbose(
                        expr.span,
                        "use a preinterned symbol instead",
                        sugg,
                        Applicability::MaybeIncorrect,
                    );
                    diag.help("add the symbol to `clippy_utils/src/sym.rs` if needed");
                },
            );
        }

        if let Some(as_str) = as_str_span(cx, expr)
            && let Node::Expr(parent) = cx.tcx.parent_hir_node(expr.hir_id)
        {
            let mut suggestions = Vec::new();

            match parent.kind {
                ExprKind::Binary(_, lhs, rhs) => {
                    suggestions.extend(self.expr_suggestion(lhs));
                    suggestions.extend(self.expr_suggestion(rhs));
                },
                ExprKind::Match(_, arms, _) => {
                    for arm in arms {
                        self.pat_suggestions(arm.pat, &mut suggestions);
                    }
                },
                _ => {},
            }

            if suggestions.is_empty() {
                return;
            }

            span_lint_and_then(
                cx,
                SYMBOL_AS_STR,
                expr.span,
                "converting a Symbol to a string",
                |diag| {
                    suggestions.push((as_str, String::new()));
                    diag.multipart_suggestion(
                        "use preinterned symbols instead",
                        suggestions,
                        Applicability::MaybeIncorrect,
                    );
                    diag.help("add the symbols to `clippy_utils/src/sym.rs` if needed");
                },
            );
        }
    }
}

/// ```ignore
/// symbol.as_str()
/// //     ^^^^^^^^
/// ```
fn as_str_span(cx: &LateContext<'_>, expr: &Expr<'_>) -> Option<Span> {
    if let ExprKind::MethodCall(_, recv, [], _) = expr.kind
        && let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
        && internal_paths::SYMBOL_AS_STR.matches(cx, method_def_id)
    {
        Some(recv.span.shrink_to_hi().to(expr.span.shrink_to_hi()))
    } else {
        None
    }
}
