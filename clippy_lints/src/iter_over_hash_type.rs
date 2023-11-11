use clippy_utils::paths::{HASHMAP_KEYS, HASHMAP_VALUES, HASHSET_ITER_TY};
use clippy_utils::{diagnostics::span_lint, match_def_path};
use clippy_utils::higher::ForLoop;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// This is a restriction lint which prevents the use of hash types (i.e., `HashSet` and `HashMap`) in for loops.
    ///
    /// ### Why is this bad?
    /// Because hash types are unordered, when iterated through such as in a for loop, the values are returned in
    /// a pseudo-random order. As a result, on redundant systems this may cause inconsistencies and anomalies.
    /// In addition, the unknown order of the elements may reduce readability or introduce other undesired
    /// side effects.
    ///
    /// ### Example
    /// ```no_run
    ///     let my_map = new Hashmap<i32, String>();
    ///     for (key, value) in my_map { /* ... */ }
    /// ```
    /// Use instead:
    /// ```no_run
    ///     let my_map = new Hashmap<i32, String>();
    ///     let mut keys = my_map.keys().clone().collect::<Vec<_>();
    ///     keys.sort();
    ///     for key in keys {
    ///         let value = &my_map[value];
    ///     }
    /// ```
    #[clippy::version = "1.75.0"]
    pub ITER_OVER_HASH_TYPE,
    restriction,
    "iterating over unordered hash-based types (`HashMap` and `HashSet`)"
}

declare_lint_pass!(IterOverHashType => [ITER_OVER_HASH_TYPE]);

impl LateLintPass<'_> for IterOverHashType {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &'_ rustc_hir::Expr<'_>) {
        if let Some(for_loop) = ForLoop::hir(expr)
            && !for_loop.body.span.from_expansion()
            && let ty = cx.typeck_results().expr_ty(for_loop.arg).peel_refs()
            && let Some(adt) = ty.ty_adt_def()
            && let did = adt.did()
            && (match_def_path(cx, did, &HASHMAP_KEYS) 
                || match_def_path(cx, did, &HASHMAP_VALUES)
                || match_def_path(cx, did, &HASHSET_ITER_TY)
                || is_type_diagnostic_item(cx, ty, sym::HashMap) 
                || is_type_diagnostic_item(cx, ty, sym::HashSet))
        {
            span_lint(
                cx,
                ITER_OVER_HASH_TYPE,
                expr.span,
                "iterating over unordered hash-based type",
            );
        };
    }
}

