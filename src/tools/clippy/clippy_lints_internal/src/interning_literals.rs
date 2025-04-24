use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::ty::match_type;
use clippy_utils::{def_path_def_ids, paths};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_lint_defs::declare_tool_lint;
use rustc_middle::mir::ConstValue;
use rustc_middle::ty;
use rustc_session::impl_lint_pass;
use rustc_span::sym;
use rustc_span::symbol::Symbol;

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

#[derive(Default)]
pub struct InterningDefinedSymbol {
    // Maps the symbol to the import path
    symbol_map: FxHashMap<u32, (&'static str, Symbol)>,
}

impl_lint_pass!(InterningDefinedSymbol => [INTERNING_LITERALS]);

impl<'tcx> LateLintPass<'tcx> for InterningDefinedSymbol {
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
            && let Some(Constant::Str(arg)) = ConstEvalCtxt::new(cx).eval_simple(arg)
        {
            span_lint_and_then(
                cx,
                INTERNING_LITERALS,
                expr.span,
                "interning a string literal",
                |diag| {
                    let value = Symbol::intern(&arg).as_u32();
                    let (message, path) = if let Some((prefix, name)) = self.symbol_map.get(&value) {
                        ("use the preinterned symbol", format!("{prefix}::{name}"))
                    } else {
                        (
                            "add the symbol to `clippy_utils/src/sym.rs` and use it",
                            format!("sym::{}", arg.replace(|ch: char| !ch.is_alphanumeric(), "_")),
                        )
                    };
                    diag.span_suggestion_verbose(expr.span, message, path, Applicability::MaybeIncorrect);
                },
            );
        }
    }
}
