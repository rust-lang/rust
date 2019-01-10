use rustc::hir;
use rustc::hir::{Body, FnDecl, Constness};
use rustc::hir::intravisit::FnKind;
// use rustc::mir::*;
use syntax::ast::{NodeId, Attribute};
use syntax_pos::Span;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_tool_lint, lint_array};
use rustc_mir::transform::qualify_min_const_fn::is_min_const_fn;
use crate::utils::{span_lint, is_entrypoint_fn};

/// **What it does:**
///
/// Suggests the use of `const` in functions and methods where possible
///
/// **Why is this bad?**
/// Not using `const` is a missed optimization. Instead of having the function execute at runtime,
/// when using `const`, it's evaluated at compiletime.
///
/// **Known problems:**
///
/// Const functions are currently still being worked on, with some features only being available
/// on nightly. This lint does not consider all edge cases currently and the suggestions may be
/// incorrect if you are using this lint on stable.
///
/// Also, the lint only runs one pass over the code. Consider these two non-const functions:
///
/// ```rust
/// fn a() -> i32 { 0 }
/// fn b() -> i32 { a() }
/// ```
///
/// When running Clippy, the lint will only suggest to make `a` const, because `b` at this time
/// can't be const as it calls a non-const function. Making `a` const and running Clippy again,
/// will suggest to make `b` const, too.
///
/// **Example:**
///
/// ```rust
/// fn new() -> Self {
///     Self {
///         random_number: 42
///     }
/// }
/// ```
///
/// Could be a const fn:
///
/// ```rust
/// const fn new() -> Self {
///     Self {
///         random_number: 42
///     }
/// }
/// ```
declare_clippy_lint! {
    pub MISSING_CONST_FOR_FN,
    nursery,
    "Lint functions definitions that could be made `const fn`"
}

#[derive(Clone)]
pub struct MissingConstForFn;

impl LintPass for MissingConstForFn {
    fn get_lints(&self) -> LintArray {
        lint_array!(MISSING_CONST_FOR_FN)
    }

    fn name(&self) -> &'static str {
        "MissingConstForFn"
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for MissingConstForFn {
    fn check_fn(
        &mut self,
        cx: &LateContext<'_, '_>,
        kind: FnKind<'_>,
        _: &FnDecl,
        _: &Body,
        span: Span,
        node_id: NodeId
    ) {
        let def_id = cx.tcx.hir().local_def_id(node_id);
        let mir = cx.tcx.optimized_mir(def_id);
        if let Err((span, err) = is_min_const_fn(cx.tcx, def_id, &mir) {
            cx.tcx.sess.span_err(span, &err);
        } else {
            match kind {
                FnKind::ItemFn(name, _generics, header, _vis, attrs) => {
                    if !can_be_const_fn(&name.as_str(), header, attrs) {
                        return;
                    }
                },
                FnKind::Method(ident, sig, _vis, attrs) => {
                    let header = sig.header;
                    let name = ident.name.as_str();
                    if !can_be_const_fn(&name, header, attrs) {
                        return;
                    }
                },
                _ => return
            }
            span_lint(cx, MISSING_CONST_FOR_FN, span, "this could be a const_fn");
        }
    }
}

fn can_be_const_fn(name: &str, header: hir::FnHeader, attrs: &[Attribute]) -> bool {
    // Main and custom entrypoints can't be `const`
    if is_entrypoint_fn(name, attrs) { return false }

    // We don't have to lint on something that's already `const`
    if header.constness == Constness::Const { return false }
    true
}
