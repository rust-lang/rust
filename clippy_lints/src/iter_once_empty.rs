use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use rustc_ast::ast::{Expr, ExprKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for usage of:
    ///
    /// - `[foo].iter()`
    /// - `[foo].iter_mut()`
    /// - `[foo].into_iter()`
    /// - `Some(foo).iter()`
    /// - `Some(foo).iter_mut()`
    /// - `Some(foo).into_iter()`
    ///
    /// ### Why is this bad?
    ///
    /// It is simpler to use the once function from the standard library:
    ///
    /// ### Example
    ///
    /// ```rust
    /// let a = [123].iter();
    /// let b = Some(123).into_iter();
    /// ```
    /// Use instead:
    /// ```rust
    /// use std::iter;
    /// let a = iter::once(&123);
    /// let b = iter::once(123);
    /// ```
    ///
    /// ### Known problems
    ///
    /// The type of the resulting iterator might become incompatible with its usage
    #[clippy::version = "1.64.0"]
    pub ITER_ONCE,
    nursery,
    "Iterator for array of length 1"
}

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for usage of:
    ///
    /// - `[].iter()`
    /// - `[].iter_mut()`
    /// - `[].into_iter()`
    /// - `None.iter()`
    /// - `None.iter_mut()`
    /// - `None.into_iter()`
    ///
    /// ### Why is this bad?
    ///
    /// It is simpler to use the empty function from the standard library:
    ///
    /// ### Example
    ///
    /// ```rust
    /// use std::{slice, option};
    /// let a: slice::Iter<i32> = [].iter();
    /// let f: option::IntoIter<i32> = None.into_iter();
    /// ```
    /// Use instead:
    /// ```rust
    /// use std::iter;
    /// let a: iter::Empty<i32> = iter::empty();
    /// let b: iter::Empty<i32> = iter::empty();
    /// ```
    ///
    /// ### Known problems
    ///
    /// The type of the resulting iterator might become incompatible with its usage
    #[clippy::version = "1.64.0"]
    pub ITER_EMPTY,
    nursery,
    "Iterator for empty array"
}

declare_lint_pass!(IterOnceEmpty => [ITER_ONCE, ITER_EMPTY]);

impl EarlyLintPass for IterOnceEmpty {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if expr.span.from_expansion() {
            // Don't lint match expressions present in
            // macro_rules! block
            return;
        }

        let (method_name, args) = if let ExprKind::MethodCall(seg, args, _) = &expr.kind {
            (seg.ident.as_str(), args)
        } else {
            return;
        };
        let arg = if args.len() == 1 {
            &args[0]
        } else {
            return;
        };

        let item = match &arg.kind {
            ExprKind::Array(v) if v.len() <= 1 => v.first(),
            ExprKind::Path(None, p) => {
                if p.segments.len() == 1 && p.segments[0].ident.name == rustc_span::sym::None {
                    None
                } else {
                    return;
                }
            },
            ExprKind::Call(f, some_args) if some_args.len() == 1 => {
                if let ExprKind::Path(None, p) = &f.kind {
                    if p.segments.len() == 1 && p.segments[0].ident.name == rustc_span::sym::Some {
                        Some(&some_args[0])
                    } else {
                        return;
                    }
                } else {
                    return;
                }
            },
            _ => return,
        };

        if let Some(i) = item {
            let (sugg, msg) = match method_name {
                "iter" => (
                    format!("std::iter::once(&{})", snippet(cx, i.span, "...")),
                    "this `iter` call can be replaced with std::iter::once",
                ),
                "iter_mut" => (
                    format!("std::iter::once(&mut {})", snippet(cx, i.span, "...")),
                    "this `iter_mut` call can be replaced with std::iter::once",
                ),
                "into_iter" => (
                    format!("std::iter::once({})", snippet(cx, i.span, "...")),
                    "this `into_iter` call can be replaced with std::iter::once",
                ),
                _ => return,
            };
            span_lint_and_sugg(cx, ITER_ONCE, expr.span, msg, "try", sugg, Applicability::Unspecified);
        } else {
            let msg = match method_name {
                "iter" => "this `iter call` can be replaced with std::iter::empty",
                "iter_mut" => "this `iter_mut` call can be replaced with std::iter::empty",
                "into_iter" => "this `into_iter` call can be replaced with std::iter::empty",
                _ => return,
            };
            span_lint_and_sugg(
                cx,
                ITER_EMPTY,
                expr.span,
                msg,
                "try",
                "std::iter::empty()".to_string(),
                Applicability::Unspecified,
            );
        }
    }
}
