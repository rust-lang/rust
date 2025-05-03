use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::ty::match_type;
use clippy_utils::{def_path_def_ids, match_def_path, paths};
use rustc_ast::LitKind;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_lint_defs::declare_tool_lint;
use rustc_middle::mir::ConstValue;
use rustc_middle::ty;
use rustc_session::impl_lint_pass;
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

impl<'tcx> LateLintPass<'tcx> for Symbols {
    fn check_crate(&mut self, cx: &LateContext<'_>) {
        let modules = [
            ("kw", &paths::KW_MODULE[..]),
            ("sym", &paths::SYM_MODULE),
            ("sym", &paths::CLIPPY_SYM_MODULE),
        ];
        for (prefix, module) in modules {
            for def_id in def_path_def_ids(cx.tcx, module) {
                // When linting `clippy_utils` itself we can't use `module_children` as it's a local def id. It will
                // still lint but the suggestion will say to add it to `sym.rs` even if it's already there
                if def_id.is_local() {
                    continue;
                }

                for item in cx.tcx.module_children(def_id) {
                    if let Res::Def(DefKind::Const, item_def_id) = item.res
                        && let ty = cx.tcx.type_of(item_def_id).instantiate_identity()
                        && match_type(cx, ty, &paths::SYMBOL)
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
            && let ExprKind::Lit(lit) = arg.kind
            && let LitKind::Str(name, _) = lit.node
        {
            span_lint_and_then(
                cx,
                INTERNING_LITERALS,
                expr.span,
                "interning a string literal",
                |diag| {
                    let (message, path) = suggestion(&mut self.symbol_map, name);
                    diag.span_suggestion_verbose(expr.span, message, path, Applicability::MaybeIncorrect);
                },
            );
        }

        if let ExprKind::Binary(_, lhs, rhs) = expr.kind {
            check_binary(cx, lhs, rhs, &mut self.symbol_map);
            check_binary(cx, rhs, lhs, &mut self.symbol_map);
        }
    }
}

fn check_binary(
    cx: &LateContext<'_>,
    lhs: &Expr<'_>,
    rhs: &Expr<'_>,
    symbols: &mut FxHashMap<u32, (&'static str, Symbol)>,
) {
    if let Some(removal_span) = as_str_span(cx, lhs)
        && let ExprKind::Lit(lit) = rhs.kind
        && let LitKind::Str(name, _) = lit.node
    {
        span_lint_and_then(cx, SYMBOL_AS_STR, lhs.span, "converting a Symbol to a string", |diag| {
            let (message, path) = suggestion(symbols, name);
            diag.multipart_suggestion_verbose(
                message,
                vec![(removal_span, String::new()), (rhs.span, path)],
                Applicability::MachineApplicable,
            );
        });
    }
}

fn suggestion(symbols: &mut FxHashMap<u32, (&'static str, Symbol)>, name: Symbol) -> (&'static str, String) {
    if let Some((prefix, name)) = symbols.get(&name.as_u32()) {
        ("use the preinterned symbol", format!("{prefix}::{name}"))
    } else {
        (
            "add the symbol to `clippy_utils/src/sym.rs` and use it",
            format!("sym::{}", name.as_str().replace(|ch: char| !ch.is_alphanumeric(), "_")),
        )
    }
}

/// ```ignore
/// symbol.as_str()
/// //     ^^^^^^^^
/// ```
fn as_str_span(cx: &LateContext<'_>, expr: &Expr<'_>) -> Option<Span> {
    if let ExprKind::MethodCall(_, recv, [], _) = expr.kind
        && let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
        && match_def_path(cx, method_def_id, &paths::SYMBOL_AS_STR)
    {
        Some(recv.span.shrink_to_hi().to(expr.span.shrink_to_hi()))
    } else {
        None
    }
}
