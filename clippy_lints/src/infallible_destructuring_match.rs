use super::utils::{get_arg_name, match_var, remove_blocks, snippet, span_lint_and_sugg};
use rustc::hir::*;
use rustc::lint::*;

/// **What it does:** Checks for matches being used to destructure a single-variant enum
/// or tuple struct where a `let` will suffice.
///
/// **Why is this bad?** Just readability – `let` doesn't nest, whereas a `match` does.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// enum Wrapper {
///     Data(i32),
/// }
///
/// let wrapper = Wrapper::Data(42);
///
/// let data = match wrapper {
///     Wrapper::Data(i) => i,
/// };
/// ```
///
/// The correct use would be:
/// ```rust
/// enum Wrapper {
///     Data(i32),
/// }
///
/// let wrapper = Wrapper::Data(42);
/// let Wrapper::Data(data) = wrapper;
/// ```
declare_clippy_lint! {
    pub INFALLIBLE_DESTRUCTURING_MATCH,
    style,
    "a match statement with a single infallible arm instead of a `let`"
}

#[derive(Copy, Clone, Default)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(INFALLIBLE_DESTRUCTURING_MATCH)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_local(&mut self, cx: &LateContext<'a, 'tcx>, local: &'tcx Local) {
        if_chain! {
            if let Some(ref expr) = local.init;
            if let Expr_::ExprMatch(ref target, ref arms, MatchSource::Normal) = expr.node;
            if arms.len() == 1 && arms[0].pats.len() == 1 && arms[0].guard.is_none();
            if let PatKind::TupleStruct(QPath::Resolved(None, ref variant_name), ref args, _) = arms[0].pats[0].node;
            if args.len() == 1;
            if let Some(arg) = get_arg_name(&args[0]);
            let body = remove_blocks(&arms[0].body);
            if match_var(body, arg);

            then {
                span_lint_and_sugg(
                    cx,
                    INFALLIBLE_DESTRUCTURING_MATCH,
                    local.span,
                    "you seem to be trying to use match to destructure a single infallible pattern. \
                     Consider using `let`",
                    "try this",
                    format!(
                        "let {}({}) = {};",
                        snippet(cx, variant_name.span, ".."),
                        snippet(cx, local.pat.span, ".."),
                        snippet(cx, target.span, ".."),
                    ),
                );
            }
        }
    }
}
