use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint;
use clippy_utils::macros::macro_backtrace;
use clippy_utils::paths::{PathNS, lookup_path_str};
use clippy_utils::{get_clippy_attr, is_from_proc_macro, sym};
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_lint::{LateContext, LateLintPass, LintContext as _, impl_lint_pass};
use rustc_middle::ty::TyCtxt;
use rustc_span::Pos as _;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to functions marked with `#[clippy::may_panic]` or configured in
    /// `may-panic-functions` that don't have a `// Panic:` comment on the line above.
    ///
    /// ### Why is this bad?
    /// Functions that may panic should be documented at their call sites to explain why the
    /// panic is acceptable or impossible in that context.
    ///
    /// ### Example
    /// ```rust,ignore
    /// #[clippy::may_panic]
    /// fn my_panicable_func(n: u32) {
    ///     if n % 2 == 0 {
    ///         panic!("even numbers are not allowed")
    ///     }
    /// }
    ///
    /// fn main() {
    ///     // Missing documentation - will lint
    ///     my_panicable_func(1);
    /// }
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// #[clippy::may_panic]
    /// fn my_panicable_func(n: u32) {
    ///     if n % 2 == 0 {
    ///         panic!("even numbers are not allowed")
    ///     }
    /// }
    ///
    /// fn main() {
    ///     // Panic: This is safe, it's an odd number
    ///     my_panicable_func(1);
    /// }
    /// ```
    ///
    /// ### Configuration
    /// This lint can be configured to check calls to external functions that may panic:
    /// ```toml
    /// # clippy.toml
    /// may-panic-functions = [
    ///     "alloc::vec::Vec::push",  # Can panic on allocation failure
    ///     "std::fs::File::open",    # Can panic in some configurations
    /// ]
    /// ```
    #[clippy::version = "1.92.0"]
    pub UNDOCUMENTED_MAY_PANIC_CALL,
    pedantic,
    "missing `// Panic:` documentation on calls to functions that may panic"
}

impl_lint_pass!(UndocumentedMayPanicCall => [UNDOCUMENTED_MAY_PANIC_CALL]);

pub struct UndocumentedMayPanicCall {
    may_panic_def_ids: FxHashSet<DefId>,
}

impl UndocumentedMayPanicCall {
    pub fn new(tcx: TyCtxt<'_>, conf: &'static Conf) -> Self {
        let may_panic_def_ids = conf
            .may_panic_functions
            .iter()
            .flat_map(|path| lookup_path_str(tcx, PathNS::Value, path))
            .collect();

        Self { may_panic_def_ids }
    }

    // A function is a may_panic_function if it has the may_panic attribute
    // or is in the may-panic-functions configuration
    fn is_may_panic_function(&self, cx: &LateContext<'_>, def_id: DefId) -> bool {
        self.may_panic_def_ids.contains(&def_id) || get_clippy_attr(cx.tcx, def_id, sym::may_panic).is_some()
    }
}

impl<'tcx> LateLintPass<'tcx> for UndocumentedMayPanicCall {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        let def_id = match &expr.kind {
            hir::ExprKind::Call(func, _args) => {
                if let hir::ExprKind::Path(qpath) = &func.kind {
                    cx.qpath_res(qpath, func.hir_id).opt_def_id()
                } else {
                    None
                }
            },
            hir::ExprKind::MethodCall(_path, _receiver, _args, _span) => {
                cx.typeck_results().type_dependent_def_id(expr.hir_id)
            },
            _ => return,
        };

        if let Some(def_id) = def_id
            && self.is_may_panic_function(cx, def_id)
            && let Some(lint_span) = check_for_missing_panic_comment(cx, expr)
        {
            span_lint(
                cx,
                UNDOCUMENTED_MAY_PANIC_CALL,
                lint_span,
                "call to a function that may panic is not documented with a `// Panic:` comment",
            );
        }
    }
}

/// Checks if a panic comment is missing and returns the span to lint at
/// Returns `None` if a panic comment exists
/// Returns `Some(span)` if a panic comment is missing
fn check_for_missing_panic_comment<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
) -> Option<rustc_span::Span> {
    let call_span = expr.span;

    if call_span.from_expansion() {
        // For external macros or proc macros, the user cannot modify the macro body,
        // so we only check callsites
        let is_external_or_proc_macro =
            call_span.in_external_macro(cx.sess().source_map()) || is_from_proc_macro(cx, expr);

        // For locally defined macros, check the macro body first before checking the callsite
        if !is_external_or_proc_macro && has_panic_comment_above_span(cx, call_span) {
            return None;
        }

        let mut lint_span = None;
        for macro_call in macro_backtrace(call_span) {
            if has_panic_comment_above_span(cx, macro_call.span) {
                return None;
            }
            lint_span = Some(macro_call.span);
        }

        lint_span
    } else if has_panic_comment_above_span(cx, call_span) {
        None
    } else {
        Some(call_span)
    }
}

/// Checks if the lines immediately preceding a span contain a "Panic:" comment
fn has_panic_comment_above_span(cx: &LateContext<'_>, span: rustc_span::Span) -> bool {
    let source_map = cx.sess().source_map();

    if let Ok(call_line) = source_map.lookup_line(span.lo())
        && call_line.line > 0
        && let Some(src) = call_line.sf.src.as_deref()
    {
        let lines = call_line.sf.lines();
        let line_starts = &lines[..=call_line.line];

        has_panic_comment_in_text(src, line_starts)
    } else {
        false
    }
}

fn has_panic_comment_in_text(src: &str, line_starts: &[rustc_span::RelativeBytePos]) -> bool {
    let mut lines = line_starts
        .array_windows::<2>()
        .rev()
        .map_while(|[start, end]| {
            let start = start.to_usize();
            let end = end.to_usize();
            let text = src.get(start..end)?;
            let trimmed = text.trim_start();
            Some((trimmed, text.len() - trimmed.len()))
        })
        .filter(|(text, _)| !text.is_empty());

    let Some((line, _)) = lines.next() else {
        return false;
    };

    if line.starts_with("//") {
        let mut current_line = line;
        loop {
            if current_line.to_ascii_uppercase().contains("PANIC:") {
                return true;
            }
            match lines.next() {
                Some((text, _)) if text.starts_with("//") => current_line = text,
                _ => return false,
            }
        }
    }

    false
}
