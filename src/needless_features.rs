//! Checks for usage of nightly features that have simple stable equivalents
//!
//! This lint is **warn** by default

use rustc::lint::*;
use rustc_front::hir::*;

use utils::span_lint;
use utils;

/// **What it does:** This lint `Warn`s on use of the `as_slice(..)` function, which is unstable.
///
/// **Why is this bad?** Using this function doesn't make your code better, but it will preclude it from building with stable Rust.
///
/// **Known problems:** None.
///
/// **Example:** `x.as_slice(..)`
declare_lint! {
    pub UNSTABLE_AS_SLICE,
    Warn,
    "as_slice is not stable and can be replaced by & v[..]\
see https://github.com/rust-lang/rust/issues/27729"
}

/// **What it does:** This lint `Warn`s on use of the `as_mut_slice(..)` function, which is unstable.
///
/// **Why is this bad?** Using this function doesn't make your code better, but it will preclude it from building with stable Rust.
///
/// **Known problems:** None.
///
/// **Example:** `x.as_mut_slice(..)`
declare_lint! {
    pub UNSTABLE_AS_MUT_SLICE,
    Warn,
    "as_mut_slice is not stable and can be replaced by &mut v[..]\
see https://github.com/rust-lang/rust/issues/27729"
}

#[derive(Copy,Clone)]
pub struct NeedlessFeaturesPass;

impl LintPass for NeedlessFeaturesPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNSTABLE_AS_SLICE, UNSTABLE_AS_MUT_SLICE)
    }
}

impl LateLintPass for NeedlessFeaturesPass {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if let ExprMethodCall(ref name, _, _) = expr.node {
            if name.node.as_str() == "as_slice" && check_paths(cx, expr) {
                span_lint(cx, UNSTABLE_AS_SLICE, expr.span,
                          "used as_slice() from the 'convert' nightly feature. Use &[..] \
                           instead");
            }
            if name.node.as_str() == "as_mut_slice" && check_paths(cx, expr) {
                span_lint(cx, UNSTABLE_AS_MUT_SLICE, expr.span,
                          "used as_mut_slice() from the 'convert' nightly feature. Use &mut [..] \
                           instead");
            }
        }
    }
}

fn check_paths(cx: &LateContext, expr: &Expr) -> bool {
    utils::match_impl_method(cx, expr, &["collections", "vec", "Vec<T>"])
}
