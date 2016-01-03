//! This LintPass catches both string addition and string addition + assignment
//!
//! Note that since we have two lints where one subsumes the other, we try to
//! disable the subsumed lint unless it has a higher level

use rustc::lint::*;
use rustc_front::hir::*;
use syntax::codemap::Spanned;

use utils::{is_exp_equal, match_type, span_lint, walk_ptrs_ty, get_parent_expr};
use utils::STRING_PATH;

/// **What it does:** This lint matches code of the form `x = x + y` (without `let`!). It is `Allow` by default.
///
/// **Why is this bad?** Because this expression needs another copy as opposed to `x.push_str(y)` (in practice LLVM will usually elide it, though). Despite [llogiq](https://github.com/llogiq)'s reservations, this lint also is `allow` by default, as some people opine that it's more readable.
///
/// **Known problems:** None. Well apart from the lint being `allow` by default. :smile:
///
/// **Example:**
///
/// ```
/// let mut x = "Hello".to_owned();
/// x = x + ", World";
/// ```
declare_lint! {
    pub STRING_ADD_ASSIGN,
    Allow,
    "using `x = x + ..` where x is a `String`; suggests using `push_str()` instead"
}

/// **What it does:** The `string_add` lint matches all instances of `x + _` where `x` is of type `String`, but only if [`string_add_assign`](#string_add_assign) does *not* match.  It is `Allow` by default.
///
/// **Why is this bad?** It's not bad in and of itself. However, this particular `Add` implementation is asymmetric (the other operand need not be `String`, but `x` does), while addition as mathematically defined is symmetric, also the `String::push_str(_)` function is a perfectly good replacement. Therefore some dislike it and wish not to have it in their code.
///
/// That said, other people think that String addition, having a long tradition in other languages is actually fine, which is why we decided to make this particular lint `allow` by default.
///
/// **Known problems:** None
///
/// **Example:**
///
/// ```
/// let x = "Hello".to_owned();
/// x + ", World"
/// ```
declare_lint! {
    pub STRING_ADD,
    Allow,
    "using `x + ..` where x is a `String`; suggests using `push_str()` instead"
}

#[derive(Copy, Clone)]
pub struct StringAdd;

impl LintPass for StringAdd {
    fn get_lints(&self) -> LintArray {
        lint_array!(STRING_ADD, STRING_ADD_ASSIGN)
    }
}

impl LateLintPass for StringAdd {
    fn check_expr(&mut self, cx: &LateContext, e: &Expr) {
        if let ExprBinary(Spanned{ node: BiAdd, .. }, ref left, _) = e.node {
            if is_string(cx, left) {
                if let Allow = cx.current_level(STRING_ADD_ASSIGN) {
                    // the string_add_assign is allow, so no duplicates
                } else {
                    let parent = get_parent_expr(cx, e);
                    if let Some(ref p) = parent {
                        if let ExprAssign(ref target, _) = p.node {
                            // avoid duplicate matches
                            if is_exp_equal(cx, target, left) { return; }
                        }
                    }
                }
                span_lint(cx, STRING_ADD, e.span,
                    "you added something to a string. \
                     Consider using `String::push_str()` instead");
            }
        } else if let ExprAssign(ref target, ref src) = e.node {
            if is_string(cx, target) && is_add(cx, src, target) {
                span_lint(cx, STRING_ADD_ASSIGN, e.span,
                    "you assigned the result of adding something to this string. \
                     Consider using `String::push_str()` instead");
            }
        }
    }
}

fn is_string(cx: &LateContext, e: &Expr) -> bool {
    match_type(cx, walk_ptrs_ty(cx.tcx.expr_ty(e)), &STRING_PATH)
}

fn is_add(cx: &LateContext, src: &Expr, target: &Expr) -> bool {
    match src.node {
        ExprBinary(Spanned{ node: BiAdd, .. }, ref left, _) =>
            is_exp_equal(cx, target, left),
        ExprBlock(ref block) => block.stmts.is_empty() &&
            block.expr.as_ref().map_or(false,
                |expr| is_add(cx, expr, target)),
        _ => false
    }
}
