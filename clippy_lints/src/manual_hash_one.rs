use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_opt;
use clippy_utils::visitors::{is_local_used, local_used_once};
use clippy_utils::{is_trait_method, path_to_local_id};
use rustc_errors::Applicability;
use rustc_hir::{BindingAnnotation, ExprKind, Local, Node, PatKind, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for cases where [`BuildHasher::hash_one`] can be used.
    ///
    /// [`BuildHasher::hash_one`]: https://doc.rust-lang.org/std/hash/trait.BuildHasher.html#method.hash_one
    ///
    /// ### Why is this bad?
    /// It is more concise to use the `hash_one` method.
    ///
    /// ### Example
    /// ```rust
    /// use std::hash::{BuildHasher, Hash, Hasher};
    /// use std::collections::hash_map::RandomState;
    ///
    /// let s = RandomState::new();
    /// let value = vec![1, 2, 3];
    ///
    /// let mut hasher = s.build_hasher();
    /// value.hash(&mut hasher);
    /// let hash = hasher.finish();
    /// ```
    /// Use instead:
    /// ```rust
    /// use std::hash::BuildHasher;
    /// use std::collections::hash_map::RandomState;
    ///
    /// let s = RandomState::new();
    /// let value = vec![1, 2, 3];
    ///
    /// let hash = s.hash_one(&value);
    /// ```
    #[clippy::version = "1.74.0"]
    pub MANUAL_HASH_ONE,
    complexity,
    "manual implementations of `BuildHasher::hash_one`"
}

pub struct ManualHashOne {
    msrv: Msrv,
}

impl ManualHashOne {
    #[must_use]
    pub fn new(msrv: Msrv) -> Self {
        Self { msrv }
    }
}

impl_lint_pass!(ManualHashOne => [MANUAL_HASH_ONE]);

impl LateLintPass<'_> for ManualHashOne {
    fn check_local(&mut self, cx: &LateContext<'_>, local: &Local<'_>) {
        // `let mut hasher = seg.build_hasher();`
        if let PatKind::Binding(BindingAnnotation::MUT, hasher, _, None) = local.pat.kind
            && let Some(init) = local.init
            && !init.span.from_expansion()
            && let ExprKind::MethodCall(seg, build_hasher, [], _) = init.kind
            && seg.ident.name == sym!(build_hasher)

            && let Node::Stmt(local_stmt) = cx.tcx.hir().get_parent(local.hir_id)
            && let Node::Block(block) = cx.tcx.hir().get_parent(local_stmt.hir_id)

            && let mut stmts = block.stmts.iter()
                .skip_while(|stmt| stmt.hir_id != local_stmt.hir_id)
                .skip(1)
                .filter(|&stmt| is_local_used(cx, stmt, hasher))

            // `hashed_value.hash(&mut hasher);`
            && let Some(hash_stmt) = stmts.next()
            && let StmtKind::Semi(hash_expr) = hash_stmt.kind
            && !hash_expr.span.from_expansion()
            && let ExprKind::MethodCall(seg, hashed_value, [ref_to_hasher], _) = hash_expr.kind
            && seg.ident.name == sym::hash
            && is_trait_method(cx, hash_expr, sym::Hash)
            && path_to_local_id(ref_to_hasher.peel_borrows(), hasher)

            && let maybe_finish_stmt = stmts.next()
            // There should be no more statements referencing `hasher`
            && stmts.next().is_none()

            // `hasher.finish()`, may be anywhere in a statement or the trailing expr of the block
            && let Some(path_expr) = local_used_once(cx, (maybe_finish_stmt, block.expr), hasher)
            && let Node::Expr(finish_expr) = cx.tcx.hir().get_parent(path_expr.hir_id)
            && !finish_expr.span.from_expansion()
            && let ExprKind::MethodCall(seg, _, [], _) = finish_expr.kind
            && seg.ident.name == sym!(finish)

            && self.msrv.meets(msrvs::BUILD_HASHER_HASH_ONE)
        {
            span_lint_hir_and_then(
                cx,
                MANUAL_HASH_ONE,
                finish_expr.hir_id,
                finish_expr.span,
                "manual implementation of `BuildHasher::hash_one`",
                |diag| {
                    if let Some(build_hasher) = snippet_opt(cx, build_hasher.span)
                        && let Some(hashed_value) = snippet_opt(cx, hashed_value.span)
                    {
                        diag.multipart_suggestion(
                            "try",
                            vec![
                                (local_stmt.span, String::new()),
                                (hash_stmt.span, String::new()),
                                (
                                    finish_expr.span,
                                    // `needless_borrows_for_generic_args` will take care of
                                    // removing the `&` when it isn't needed
                                    format!("{build_hasher}.hash_one(&{hashed_value})")
                                )
                            ],
                            Applicability::MachineApplicable,
                        );

                    }
                },
            );
        }
    }

    extract_msrv_attr!(LateContext);
}
