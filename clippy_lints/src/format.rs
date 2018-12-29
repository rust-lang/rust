// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::utils::paths;
use crate::utils::{
    in_macro, is_expn_of, last_path_segment, match_def_path, match_type, opt_def_id, resolve_node, snippet,
    span_lint_and_then, walk_ptrs_ty,
};
use if_chain::if_chain;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::ty;
use rustc::{declare_tool_lint, lint_array};
use rustc_errors::Applicability;
use syntax::ast::LitKind;

/// **What it does:** Checks for the use of `format!("string literal with no
/// argument")` and `format!("{}", foo)` where `foo` is a string.
///
/// **Why is this bad?** There is no point of doing that. `format!("foo")` can
/// be replaced by `"foo".to_owned()` if you really need a `String`. The even
/// worse `&format!("foo")` is often encountered in the wild. `format!("{}",
/// foo)` can be replaced by `foo.clone()` if `foo: String` or `foo.to_owned()`
/// if `foo: &str`.
///
/// **Known problems:** None.
///
/// **Examples:**
/// ```rust
/// format!("foo")
/// format!("{}", foo)
/// ```
declare_clippy_lint! {
    pub USELESS_FORMAT,
    complexity,
    "useless use of `format!`"
}

#[derive(Copy, Clone, Debug)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array![USELESS_FORMAT]
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let Some(span) = is_expn_of(expr.span, "format") {
            if in_macro(span) {
                return;
            }
            match expr.node {
                // `format!("{}", foo)` expansion
                ExprKind::Call(ref fun, ref args) => {
                    if_chain! {
                        if let ExprKind::Path(ref qpath) = fun.node;
                        if args.len() == 3;
                        if let Some(fun_def_id) = opt_def_id(resolve_node(cx, qpath, fun.hir_id));
                        if match_def_path(cx.tcx, fun_def_id, &paths::FMT_ARGUMENTS_NEWV1FORMATTED);
                        if check_single_piece(&args[0]);
                        if let Some(format_arg) = get_single_string_arg(cx, &args[1]);
                        if check_unformatted(&args[2]);
                        if let ExprKind::AddrOf(_, ref format_arg) = format_arg.node;
                        then {
                            let (message, sugg) = if_chain! {
                                if let ExprKind::MethodCall(ref path, _, _) = format_arg.node;
                                if path.ident.as_interned_str() == "to_string";
                                then {
                                    ("`to_string()` is enough",
                                    snippet(cx, format_arg.span, "<arg>").to_string())
                                } else {
                                    ("consider using .to_string()",
                                    format!("{}.to_string()", snippet(cx, format_arg.span, "<arg>")))
                                }
                            };

                            span_lint_and_then(cx, USELESS_FORMAT, span, "useless use of `format!`", |db| {
                                db.span_suggestion_with_applicability(
                                    expr.span,
                                    message,
                                    sugg,
                                    Applicability::MachineApplicable,
                                );
                            });
                        }
                    }
                },
                // `format!("foo")` expansion contains `match () { () => [], }`
                ExprKind::Match(ref matchee, _, _) => {
                    if let ExprKind::Tup(ref tup) = matchee.node {
                        if tup.is_empty() {
                            let sugg = format!("{}.to_string()", snippet(cx, expr.span, "<expr>").into_owned());
                            span_lint_and_then(cx, USELESS_FORMAT, span, "useless use of `format!`", |db| {
                                db.span_suggestion_with_applicability(
                                    span,
                                    "consider using .to_string()",
                                    sugg,
                                    Applicability::MachineApplicable, // snippet
                                );
                            });
                        }
                    }
                },
                _ => (),
            }
        }
    }
}

/// Checks if the expressions matches `&[""]`
fn check_single_piece(expr: &Expr) -> bool {
    if_chain! {
        if let ExprKind::AddrOf(_, ref expr) = expr.node; // &[""]
        if let ExprKind::Array(ref exprs) = expr.node; // [""]
        if exprs.len() == 1;
        if let ExprKind::Lit(ref lit) = exprs[0].node;
        if let LitKind::Str(ref lit, _) = lit.node;
        then {
            return lit.as_str().is_empty();
        }
    }

    false
}

/// Checks if the expressions matches
/// ```rust,ignore
/// &match (&"arg",) {
/// (__arg0,) => [::std::fmt::ArgumentV1::new(__arg0,
/// ::std::fmt::Display::fmt)],
/// }
/// ```
/// and that the type of `__arg0` is `&str` or `String`,
/// then returns the span of first element of the matched tuple.
fn get_single_string_arg<'a>(cx: &LateContext<'_, '_>, expr: &'a Expr) -> Option<&'a Expr> {
    if_chain! {
        if let ExprKind::AddrOf(_, ref expr) = expr.node;
        if let ExprKind::Match(ref match_expr, ref arms, _) = expr.node;
        if arms.len() == 1;
        if arms[0].pats.len() == 1;
        if let PatKind::Tuple(ref pat, None) = arms[0].pats[0].node;
        if pat.len() == 1;
        if let ExprKind::Array(ref exprs) = arms[0].body.node;
        if exprs.len() == 1;
        if let ExprKind::Call(_, ref args) = exprs[0].node;
        if args.len() == 2;
        if let ExprKind::Path(ref qpath) = args[1].node;
        if let Some(fun_def_id) = opt_def_id(resolve_node(cx, qpath, args[1].hir_id));
        if match_def_path(cx.tcx, fun_def_id, &paths::DISPLAY_FMT_METHOD);
        then {
            let ty = walk_ptrs_ty(cx.tables.pat_ty(&pat[0]));
            if ty.sty == ty::Str || match_type(cx, ty, &paths::STRING) {
                if let ExprKind::Tup(ref values) = match_expr.node {
                    return Some(&values[0]);
                }
            }
        }
    }

    None
}

/// Checks if the expression matches
/// ```rust,ignore
/// &[_ {
///    format: _ {
///         width: _::Implied,
///         ...
///    },
///    ...,
/// }]
/// ```
fn check_unformatted(expr: &Expr) -> bool {
    if_chain! {
        if let ExprKind::AddrOf(_, ref expr) = expr.node;
        if let ExprKind::Array(ref exprs) = expr.node;
        if exprs.len() == 1;
        if let ExprKind::Struct(_, ref fields, _) = exprs[0].node;
        if let Some(format_field) = fields.iter().find(|f| f.ident.name == "format");
        if let ExprKind::Struct(_, ref fields, _) = format_field.expr.node;
        if let Some(width_field) = fields.iter().find(|f| f.ident.name == "width");
        if let ExprKind::Path(ref width_qpath) = width_field.expr.node;
        if last_path_segment(width_qpath).ident.name == "Implied";
        if let Some(precision_field) = fields.iter().find(|f| f.ident.name == "precision");
        if let ExprKind::Path(ref precision_path) = precision_field.expr.node;
        if last_path_segment(precision_path).ident.name == "Implied";
        then {
            return true;
        }
    }

    false
}
