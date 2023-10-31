use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::ty::match_type;
use clippy_utils::{is_diag_item_method, is_trait_method, match_def_path, path_to_local_id, paths};
use rustc_errors::Applicability;
use rustc_hir::{Body, Closure, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `lines.filter_map(Result::ok)` or `lines.flat_map(Result::ok)`
    /// when `lines` has type `std::io::Lines`.
    ///
    /// ### Why is this bad?
    /// `Lines` instances might produce a never-ending stream of `Err`, in which case
    /// `filter_map(Result::ok)` will enter an infinite loop while waiting for an
    /// `Ok` variant. Calling `next()` once is sufficient to enter the infinite loop,
    /// even in the absence of explicit loops in the user code.
    ///
    /// This situation can arise when working with user-provided paths. On some platforms,
    /// `std::fs::File::open(path)` might return `Ok(fs)` even when `path` is a directory,
    /// but any later attempt to read from `fs` will return an error.
    ///
    /// ### Known problems
    /// This lint suggests replacing `filter_map()` or `flat_map()` applied to a `Lines`
    /// instance in all cases. There are two cases where the suggestion might not be
    /// appropriate or necessary:
    ///
    /// - If the `Lines` instance can never produce any error, or if an error is produced
    ///   only once just before terminating the iterator, using `map_while()` is not
    ///   necessary but will not do any harm.
    /// - If the `Lines` instance can produce intermittent errors then recover and produce
    ///   successful results, using `map_while()` would stop at the first error.
    ///
    /// ### Example
    /// ```rust
    /// # use std::{fs::File, io::{self, BufRead, BufReader}};
    /// # let _ = || -> io::Result<()> {
    /// let mut lines = BufReader::new(File::open("some-path")?).lines().filter_map(Result::ok);
    /// // If "some-path" points to a directory, the next statement never terminates:
    /// let first_line: Option<String> = lines.next();
    /// # Ok(()) };
    /// ```
    /// Use instead:
    /// ```rust
    /// # use std::{fs::File, io::{self, BufRead, BufReader}};
    /// # let _ = || -> io::Result<()> {
    /// let mut lines = BufReader::new(File::open("some-path")?).lines().map_while(Result::ok);
    /// let first_line: Option<String> = lines.next();
    /// # Ok(()) };
    /// ```
    #[clippy::version = "1.70.0"]
    pub LINES_FILTER_MAP_OK,
    suspicious,
    "filtering `std::io::Lines` with `filter_map()` or `flat_map()` might cause an infinite loop"
}
declare_lint_pass!(LinesFilterMapOk => [LINES_FILTER_MAP_OK]);

impl LateLintPass<'_> for LinesFilterMapOk {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if let ExprKind::MethodCall(fm_method, fm_receiver, [fm_arg], fm_span) = expr.kind &&
            is_trait_method(cx, expr, sym::Iterator) &&
            (fm_method.ident.as_str() == "filter_map" || fm_method.ident.as_str() == "flat_map") &&
            match_type(cx, cx.typeck_results().expr_ty_adjusted(fm_receiver), &paths::STD_IO_LINES)
        {
            let lint = match &fm_arg.kind {
                // Detect `Result::ok`
                ExprKind::Path(qpath) =>
                    cx.qpath_res(qpath, fm_arg.hir_id).opt_def_id().map(|did|
                        match_def_path(cx, did, &paths::CORE_RESULT_OK_METHOD)).unwrap_or_default(),
                // Detect `|x| x.ok()`
                ExprKind::Closure(Closure { body, .. }) =>
                    if let Body { params: [param], value, .. } = cx.tcx.hir().body(*body) &&
                        let ExprKind::MethodCall(method, receiver, [], _) = value.kind &&
                        path_to_local_id(receiver, param.pat.hir_id) &&
                        let Some(method_did) = cx.typeck_results().type_dependent_def_id(value.hir_id)
                    {
                        is_diag_item_method(cx, method_did, sym::Result) && method.ident.as_str() == "ok"
                    } else {
                        false
                    }
                _ => false,
            };
            if lint {
                span_lint_and_then(cx,
                    LINES_FILTER_MAP_OK,
                    fm_span,
                    &format!("`{}()` will run forever if the iterator repeatedly produces an `Err`", fm_method.ident),
                    |diag| {
                        diag.span_note(
                            fm_receiver.span,
                            "this expression returning a `std::io::Lines` may produce an infinite number of `Err` in case of a read error");
                        diag.span_suggestion(fm_span, "replace with", "map_while(Result::ok)", Applicability::MaybeIncorrect);
                    });
                }
        }
    }
}
