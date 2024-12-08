use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::ty::match_type;
use clippy_utils::{def_path_def_ids, is_expn_of, match_def_path, paths};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::{BinOpKind, Expr, ExprKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::mir::ConstValue;
use rustc_middle::ty;
use rustc_session::impl_lint_pass;
use rustc_span::sym;
use rustc_span::symbol::Symbol;

use std::borrow::Cow;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for interning symbols that have already been pre-interned and defined as constants.
    ///
    /// ### Why is this bad?
    /// It's faster and easier to use the symbol constant.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let _ = sym!(f32);
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// let _ = sym::f32;
    /// ```
    pub INTERNING_DEFINED_SYMBOL,
    internal,
    "interning a symbol that is pre-interned and defined as a constant"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for unnecessary conversion from Symbol to a string.
    ///
    /// ### Why is this bad?
    /// It's faster use symbols directly instead of strings.
    ///
    /// ### Example
    /// ```rust,ignore
    /// symbol.as_str() == "clippy";
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// symbol == sym::clippy;
    /// ```
    pub UNNECESSARY_SYMBOL_STR,
    internal,
    "unnecessary conversion between Symbol and string"
}

#[derive(Default)]
pub struct InterningDefinedSymbol {
    // Maps the symbol value to the constant DefId.
    symbol_map: FxHashMap<u32, DefId>,
}

impl_lint_pass!(InterningDefinedSymbol => [INTERNING_DEFINED_SYMBOL, UNNECESSARY_SYMBOL_STR]);

impl<'tcx> LateLintPass<'tcx> for InterningDefinedSymbol {
    fn check_crate(&mut self, cx: &LateContext<'_>) {
        if !self.symbol_map.is_empty() {
            return;
        }

        for &module in &[&paths::KW_MODULE, &paths::SYM_MODULE] {
            for def_id in def_path_def_ids(cx.tcx, module) {
                for item in cx.tcx.module_children(def_id) {
                    if let Res::Def(DefKind::Const, item_def_id) = item.res
                        && let ty = cx.tcx.type_of(item_def_id).instantiate_identity()
                        && match_type(cx, ty, &paths::SYMBOL)
                        && let Ok(ConstValue::Scalar(value)) = cx.tcx.const_eval_poly(item_def_id)
                        && let Some(value) = value.to_u32().discard_err()
                    {
                        self.symbol_map.insert(value, item_def_id);
                    }
                }
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Call(func, [arg]) = &expr.kind
            && let ty::FnDef(def_id, _) = cx.typeck_results().expr_ty(func).kind()
            && match_def_path(cx, *def_id, &paths::SYMBOL_INTERN)
            && let Some(Constant::Str(arg)) = ConstEvalCtxt::new(cx).eval_simple(arg)
            && let value = Symbol::intern(&arg).as_u32()
            && let Some(&def_id) = self.symbol_map.get(&value)
        {
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
        static IDENT_STR_PATHS: &[&[&str]] = &[&paths::IDENT_AS_STR];
        static SYMBOL_STR_PATHS: &[&[&str]] = &[&paths::SYMBOL_AS_STR, &paths::SYMBOL_TO_IDENT_STRING];
        let call = if let ExprKind::AddrOf(_, _, e) = expr.kind
            && let ExprKind::Unary(UnOp::Deref, e) = e.kind
        {
            e
        } else {
            expr
        };
        if let ExprKind::MethodCall(_, item, [], _) = call.kind
            // is a method call
            && let Some(did) = cx.typeck_results().type_dependent_def_id(call.hir_id)
            && let ty = cx.typeck_results().expr_ty(item)
            // ...on either an Ident or a Symbol
            && let Some(is_ident) = if match_type(cx, ty, &paths::SYMBOL) {
                Some(false)
            } else if match_type(cx, ty, &paths::IDENT) {
                Some(true)
            } else {
                None
            }
            // ...which converts it to a string
            && let paths = if is_ident { IDENT_STR_PATHS } else { SYMBOL_STR_PATHS }
            && let Some(is_to_owned) = paths
                .iter()
                .find_map(|path| if match_def_path(cx, did, path) {
                    Some(path == &paths::SYMBOL_TO_IDENT_STRING)
                } else {
                    None
                })
                .or_else(|| if cx.tcx.is_diagnostic_item(sym::to_string_method, did) {
                    Some(true)
                } else {
                    None
                })
        {
            return Some(SymbolStrExpr::Expr {
                item,
                is_ident,
                is_to_owned,
            });
        }
        // is a string constant
        if let Some(Constant::Str(s)) = ConstEvalCtxt::new(cx).eval_simple(expr) {
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
