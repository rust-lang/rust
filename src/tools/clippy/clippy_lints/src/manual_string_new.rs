use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_ast::LitKind;
use rustc_errors::Applicability::MachineApplicable;
use rustc_hir::{Expr, ExprKind, PathSegment, QPath, TyKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{sym, symbol, Span};

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for usage of `""` to create a `String`, such as `"".to_string()`, `"".to_owned()`,
    /// `String::from("")` and others.
    ///
    /// ### Why is this bad?
    ///
    /// Different ways of creating an empty string makes your code less standardized, which can
    /// be confusing.
    ///
    /// ### Example
    /// ```rust
    /// let a = "".to_string();
    /// let b: String = "".into();
    /// ```
    /// Use instead:
    /// ```rust
    /// let a = String::new();
    /// let b = String::new();
    /// ```
    #[clippy::version = "1.65.0"]
    pub MANUAL_STRING_NEW,
    pedantic,
    "empty String is being created manually"
}
declare_lint_pass!(ManualStringNew => [MANUAL_STRING_NEW]);

impl LateLintPass<'_> for ManualStringNew {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }

        let ty = cx.typeck_results().expr_ty(expr);
        match ty.kind() {
            ty::Adt(adt_def, _) if adt_def.is_struct() => {
                if cx.tcx.lang_items().string() != Some(adt_def.did()) {
                    return;
                }
            },
            _ => return,
        }

        match expr.kind {
            ExprKind::Call(func, args) => {
                parse_call(cx, expr.span, func, args);
            },
            ExprKind::MethodCall(path_segment, receiver, ..) => {
                parse_method_call(cx, expr.span, path_segment, receiver);
            },
            _ => (),
        }
    }
}

/// Checks if an expression's kind corresponds to an empty &str.
fn is_expr_kind_empty_str(expr_kind: &ExprKind<'_>) -> bool {
    if  let ExprKind::Lit(lit) = expr_kind &&
        let LitKind::Str(value, _) = lit.node &&
        value == symbol::kw::Empty
    {
        return true;
    }

    false
}

fn warn_then_suggest(cx: &LateContext<'_>, span: Span) {
    span_lint_and_sugg(
        cx,
        MANUAL_STRING_NEW,
        span,
        "empty String is being created manually",
        "consider using",
        "String::new()".into(),
        MachineApplicable,
    );
}

/// Tries to parse an expression as a method call, emitting the warning if necessary.
fn parse_method_call(cx: &LateContext<'_>, span: Span, path_segment: &PathSegment<'_>, receiver: &Expr<'_>) {
    let ident = path_segment.ident.as_str();
    let method_arg_kind = &receiver.kind;
    if ["to_string", "to_owned", "into"].contains(&ident) && is_expr_kind_empty_str(method_arg_kind) {
        warn_then_suggest(cx, span);
    } else if let ExprKind::Call(func, args) = method_arg_kind {
        // If our first argument is a function call itself, it could be an `unwrap`-like function.
        // E.g. String::try_from("hello").unwrap(), TryFrom::try_from("").expect("hello"), etc.
        parse_call(cx, span, func, args);
    }
}

/// Tries to parse an expression as a function call, emitting the warning if necessary.
fn parse_call(cx: &LateContext<'_>, span: Span, func: &Expr<'_>, args: &[Expr<'_>]) {
    if args.len() != 1 {
        return;
    }

    let arg_kind = &args[0].kind;
    if let ExprKind::Path(qpath) = &func.kind {
        if let QPath::TypeRelative(_, _) = qpath {
            // String::from(...) or String::try_from(...)
            if  let QPath::TypeRelative(ty, path_seg) = qpath &&
                [sym::from, sym::try_from].contains(&path_seg.ident.name) &&
                let TyKind::Path(qpath) = &ty.kind &&
                let QPath::Resolved(_, path) = qpath &&
                let [path_seg] = path.segments &&
                path_seg.ident.name == sym::String &&
                is_expr_kind_empty_str(arg_kind)
            {
                warn_then_suggest(cx, span);
            }
        } else if let QPath::Resolved(_, path) = qpath {
            // From::from(...) or TryFrom::try_from(...)
            if  let [path_seg1, path_seg2] = path.segments &&
                is_expr_kind_empty_str(arg_kind) && (
                    (path_seg1.ident.name == sym::From && path_seg2.ident.name == sym::from) ||
                    (path_seg1.ident.name == sym::TryFrom && path_seg2.ident.name == sym::try_from)
                )
            {
                warn_then_suggest(cx, span);
            }
        }
    }
}
