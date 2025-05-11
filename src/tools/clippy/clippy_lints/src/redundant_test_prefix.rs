use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_test_function;
use clippy_utils::visitors::for_each_expr;
use rustc_errors::Applicability;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{self as hir, Body, ExprKind, FnDecl};
use rustc_lexer::is_ident;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::def_id::LocalDefId;
use rustc_span::{Span, Symbol, edition};
use std::borrow::Cow;
use std::ops::ControlFlow;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for test functions (functions annotated with `#[test]`) that are prefixed
    /// with `test_` which is redundant.
    ///
    /// ### Why is this bad?
    /// This is redundant because test functions are already annotated with `#[test]`.
    /// Moreover, it clutters the output of `cargo test` since test functions are expanded as
    /// `module::tests::test_use_case` in the output. Without the redundant prefix, the output
    /// becomes `module::tests::use_case`, which is more readable.
    ///
    /// ### Example
    /// ```no_run
    /// #[cfg(test)]
    /// mod tests {
    ///   use super::*;
    ///
    ///   #[test]
    ///   fn test_use_case() {
    ///       // test code
    ///   }
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// #[cfg(test)]
    /// mod tests {
    ///   use super::*;
    ///
    ///   #[test]
    ///   fn use_case() {
    ///       // test code
    ///   }
    /// }
    /// ```
    #[clippy::version = "1.88.0"]
    pub REDUNDANT_TEST_PREFIX,
    restriction,
    "redundant `test_` prefix in test function name"
}

declare_lint_pass!(RedundantTestPrefix => [REDUNDANT_TEST_PREFIX]);

impl<'tcx> LateLintPass<'tcx> for RedundantTestPrefix {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'_>,
        _decl: &FnDecl<'_>,
        body: &'tcx Body<'_>,
        _span: Span,
        fn_def_id: LocalDefId,
    ) {
        // Ignore methods and closures.
        let FnKind::ItemFn(ref ident, ..) = kind else {
            return;
        };

        // Skip the lint if the function is within a macro expansion.
        if ident.span.from_expansion() {
            return;
        }

        // Skip if the function name does not start with `test_`.
        if !ident.as_str().starts_with("test_") {
            return;
        }

        // If the function is not a test function, skip the lint.
        if !is_test_function(cx.tcx, fn_def_id) {
            return;
        }

        span_lint_and_then(
            cx,
            REDUNDANT_TEST_PREFIX,
            ident.span,
            "redundant `test_` prefix in test function name",
            |diag| {
                let non_prefixed = Symbol::intern(ident.as_str().trim_start_matches("test_"));
                if is_invalid_ident(non_prefixed) {
                    // If the prefix-trimmed name is not a valid function name, do not provide an
                    // automatic fix, just suggest renaming the function.
                    diag.help(
                        "consider function renaming (just removing `test_` prefix will produce invalid function name)",
                    );
                } else {
                    let (sugg, msg): (Cow<'_, str>, _) = if name_conflicts(cx, body, non_prefixed) {
                        // If `non_prefixed` conflicts with another function in the same module/scope,
                        // do not provide an automatic fix, but still emit a fix suggestion.
                        (
                            format!("{non_prefixed}_works").into(),
                            "consider function renaming (just removing `test_` prefix will cause a name conflict)",
                        )
                    } else {
                        // If `non_prefixed` is a valid identifier and does not conflict with another function,
                        // so we can suggest an auto-fix.
                        (non_prefixed.as_str().into(), "consider removing the `test_` prefix")
                    };
                    diag.span_suggestion(ident.span, msg, sugg, Applicability::MaybeIncorrect);
                }
            },
        );
    }
}

/// Checks whether removal of the `_test` prefix from the function name will cause a name conflict.
///
/// There should be no other function with the same name in the same module/scope. Also, there
/// should not be any function call with the same name within the body of the function, to avoid
/// recursion.
fn name_conflicts<'tcx>(cx: &LateContext<'tcx>, body: &'tcx Body<'_>, fn_name: Symbol) -> bool {
    let tcx = cx.tcx;
    let id = body.id().hir_id;

    // Iterate over items in the same module/scope
    let (module, _module_span, _module_hir) = tcx.hir_get_module(tcx.parent_module(id));
    if module
        .item_ids
        .iter()
        .any(|item| matches!(tcx.hir_item(*item).kind, hir::ItemKind::Fn { ident, .. } if ident.name == fn_name))
    {
        // Name conflict found
        return true;
    }

    // Also check that within the body of the function there is also no function call
    // with the same name (since it will result in recursion)
    for_each_expr(cx, body, |expr| {
        if let ExprKind::Path(qpath) = &expr.kind
            && let Some(def_id) = cx.qpath_res(qpath, expr.hir_id).opt_def_id()
            && let Some(name) = tcx.opt_item_name(def_id)
            && name == fn_name
        {
            // Function call with the same name found
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .is_some()
}

fn is_invalid_ident(ident: Symbol) -> bool {
    // The identifier is either a reserved keyword, or starts with an invalid sequence.
    ident.is_reserved(|| edition::LATEST_STABLE_EDITION) || !is_ident(ident.as_str())
}
