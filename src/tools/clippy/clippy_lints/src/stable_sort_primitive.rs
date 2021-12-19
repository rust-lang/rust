use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::{is_slice_of_primitives, sugg::Sugg};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// When sorting primitive values (integers, bools, chars, as well
    /// as arrays, slices, and tuples of such items), it is better to
    /// use an unstable sort than a stable sort.
    ///
    /// ### Why is this bad?
    /// Using a stable sort consumes more memory and cpu cycles. Because
    /// values which compare equal are identical, preserving their
    /// relative order (the guarantee that a stable sort provides) means
    /// nothing, while the extra costs still apply.
    ///
    /// ### Example
    /// ```rust
    /// let mut vec = vec![2, 1, 3];
    /// vec.sort();
    /// ```
    /// Use instead:
    /// ```rust
    /// let mut vec = vec![2, 1, 3];
    /// vec.sort_unstable();
    /// ```
    #[clippy::version = "1.47.0"]
    pub STABLE_SORT_PRIMITIVE,
    perf,
    "use of sort() when sort_unstable() is equivalent"
}

declare_lint_pass!(StableSortPrimitive => [STABLE_SORT_PRIMITIVE]);

/// The three "kinds" of sorts
enum SortingKind {
    Vanilla,
    /* The other kinds of lint are currently commented out because they
     * can map distinct values to equal ones. If the key function is
     * provably one-to-one, or if the Cmp function conserves equality,
     * then they could be linted on, but I don't know if we can check
     * for that. */

    /* ByKey,
     * ByCmp, */
}
impl SortingKind {
    /// The name of the stable version of this kind of sort
    fn stable_name(&self) -> &str {
        match self {
            SortingKind::Vanilla => "sort",
            /* SortingKind::ByKey => "sort_by_key",
             * SortingKind::ByCmp => "sort_by", */
        }
    }
    /// The name of the unstable version of this kind of sort
    fn unstable_name(&self) -> &str {
        match self {
            SortingKind::Vanilla => "sort_unstable",
            /* SortingKind::ByKey => "sort_unstable_by_key",
             * SortingKind::ByCmp => "sort_unstable_by", */
        }
    }
    /// Takes the name of a function call and returns the kind of sort
    /// that corresponds to that function name (or None if it isn't)
    fn from_stable_name(name: &str) -> Option<SortingKind> {
        match name {
            "sort" => Some(SortingKind::Vanilla),
            // "sort_by" => Some(SortingKind::ByCmp),
            // "sort_by_key" => Some(SortingKind::ByKey),
            _ => None,
        }
    }
}

/// A detected instance of this lint
struct LintDetection {
    slice_name: String,
    method: SortingKind,
    method_args: String,
    slice_type: String,
}

fn detect_stable_sort_primitive(cx: &LateContext<'_>, expr: &Expr<'_>) -> Option<LintDetection> {
    if_chain! {
        if let ExprKind::MethodCall(method_name, _, args, _) = &expr.kind;
        if let Some(slice) = &args.get(0);
        if let Some(method) = SortingKind::from_stable_name(method_name.ident.name.as_str());
        if let Some(slice_type) = is_slice_of_primitives(cx, slice);
        then {
            let args_str = args.iter().skip(1).map(|arg| Sugg::hir(cx, arg, "..").to_string()).collect::<Vec<String>>().join(", ");
            Some(LintDetection { slice_name: Sugg::hir(cx, slice, "..").to_string(), method, method_args: args_str, slice_type })
        } else {
            None
        }
    }
}

impl LateLintPass<'_> for StableSortPrimitive {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if let Some(detection) = detect_stable_sort_primitive(cx, expr) {
            span_lint_and_then(
                cx,
                STABLE_SORT_PRIMITIVE,
                expr.span,
                format!(
                    "used `{}` on primitive type `{}`",
                    detection.method.stable_name(),
                    detection.slice_type,
                )
                .as_str(),
                |diag| {
                    diag.span_suggestion(
                        expr.span,
                        "try",
                        format!(
                            "{}.{}({})",
                            detection.slice_name,
                            detection.method.unstable_name(),
                            detection.method_args,
                        ),
                        Applicability::MachineApplicable,
                    );
                    diag.note(
                        "an unstable sort would perform faster without any observable difference for this data type",
                    );
                },
            );
        }
    }
}
