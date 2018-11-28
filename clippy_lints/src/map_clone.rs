// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::rustc::hir;
use crate::rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use crate::rustc::{declare_tool_lint, lint_array};
use crate::rustc_errors::Applicability;
use crate::syntax::ast::Ident;
use crate::syntax::source_map::Span;
use crate::utils::paths;
use crate::utils::{
    in_macro, match_trait_method, match_type, remove_blocks, snippet_with_applicability, span_lint_and_sugg,
};
use if_chain::if_chain;

#[derive(Clone)]
pub struct Pass;

/// **What it does:** Checks for usage of `iterator.map(|x| x.clone())` and suggests
/// `iterator.cloned()` instead
///
/// **Why is this bad?** Readability, this can be written more concisely
///
/// **Known problems:** None.
///
/// **Example:**
///
/// ```rust
/// let x = vec![42, 43];
/// let y = x.iter();
/// let z = y.map(|i| *i);
/// ```
///
/// The correct use would be:
///
/// ```rust
/// let x = vec![42, 43];
/// let y = x.iter();
/// let z = y.cloned();
/// ```
declare_clippy_lint! {
    pub MAP_CLONE,
    style,
    "using `iterator.map(|x| x.clone())`, or dereferencing closures for `Copy` types"
}

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(MAP_CLONE)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'_, '_>, e: &hir::Expr) {
        if in_macro(e.span) {
            return;
        }

        if_chain! {
            if let hir::ExprKind::MethodCall(ref method, _, ref args) = e.node;
            if args.len() == 2;
            if method.ident.as_str() == "map";
            let ty = cx.tables.expr_ty(&args[0]);
            if match_type(cx, ty, &paths::OPTION) || match_trait_method(cx, e, &paths::ITERATOR);
            if let hir::ExprKind::Closure(_, _, body_id, _, _) = args[1].node;
            let closure_body = cx.tcx.hir.body(body_id);
            let closure_expr = remove_blocks(&closure_body.value);
            then {
                match closure_body.arguments[0].pat.node {
                    hir::PatKind::Ref(ref inner, _) => if let hir::PatKind::Binding(
                        hir::BindingAnnotation::Unannotated, _, name, None
                    ) = inner.node {
                        lint(cx, e.span, args[0].span, name, closure_expr);
                    },
                    hir::PatKind::Binding(hir::BindingAnnotation::Unannotated, _, name, None) => {
                        match closure_expr.node {
                            hir::ExprKind::Unary(hir::UnOp::UnDeref, ref inner) => {
                                if !cx.tables.expr_ty(inner).is_box() {
                                    lint(cx, e.span, args[0].span, name, inner);
                                }
                            },
                            hir::ExprKind::MethodCall(ref method, _, ref obj) => {
                                if method.ident.as_str() == "clone"
                                    && match_trait_method(cx, closure_expr, &paths::CLONE_TRAIT) {
                                    lint(cx, e.span, args[0].span, name, &obj[0]);
                                }
                            },
                            _ => {},
                        }
                    },
                    _ => {},
                }
            }
        }
    }
}

fn lint(cx: &LateContext<'_, '_>, replace: Span, root: Span, name: Ident, path: &hir::Expr) {
    if let hir::ExprKind::Path(hir::QPath::Resolved(None, ref path)) = path.node {
        if path.segments.len() == 1 && path.segments[0].ident == name {
            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                MAP_CLONE,
                replace,
                "You are using an explicit closure for cloning elements",
                "Consider calling the dedicated `cloned` method",
                format!(
                    "{}.cloned()",
                    snippet_with_applicability(cx, root, "..", &mut applicability)
                ),
                applicability,
            )
        }
    }
}
