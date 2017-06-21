use rustc::hir::*;
use rustc::lint::*;
use syntax::codemap::Spanned;
use utils::SpanlessEq;
use utils::{match_type, paths, span_lint, span_lint_and_sugg, walk_ptrs_ty, get_parent_expr};

/// **What it does:** Checks for string appends of the form `x = x + y` (without
/// `let`!).
///
/// **Why is this bad?** It's not really bad, but some people think that the
/// `.push_str(_)` method is more readable.
///
/// **Known problems:** None.
///
/// **Example:**
///
/// ```rust
/// let mut x = "Hello".to_owned();
/// x = x + ", World";
/// ```
declare_lint! {
    pub STRING_ADD_ASSIGN,
    Allow,
    "using `x = x + ..` where x is a `String` instead of `push_str()`"
}

/// **What it does:** Checks for all instances of `x + _` where `x` is of type
/// `String`, but only if [`string_add_assign`](#string_add_assign) does *not*
/// match.
///
/// **Why is this bad?** It's not bad in and of itself. However, this particular
/// `Add` implementation is asymmetric (the other operand need not be `String`,
/// but `x` does), while addition as mathematically defined is symmetric, also
/// the `String::push_str(_)` function is a perfectly good replacement.
/// Therefore some dislike it and wish not to have it in their code.
///
/// That said, other people think that string addition, having a long tradition
/// in other languages is actually fine, which is why we decided to make this
/// particular lint `allow` by default.
///
/// **Known problems:** None.
///
/// **Example:**
///
/// ```rust
/// let x = "Hello".to_owned();
/// x + ", World"
/// ```
declare_lint! {
    pub STRING_ADD,
    Allow,
    "using `x + ..` where x is a `String` instead of `push_str()`"
}

/// **What it does:** Checks for the `as_bytes` method called on string literals
/// that contain only ASCII characters.
///
/// **Why is this bad?** Byte string literals (e.g. `b"foo"`) can be used
/// instead. They are shorter but less discoverable than `as_bytes()`.
///
/// **Known Problems:** None.
///
/// **Example:**
/// ```rust
/// let bs = "a byte string".as_bytes();
/// ```
declare_lint! {
    pub STRING_LIT_AS_BYTES,
    Warn,
    "calling `as_bytes` on a string literal instead of using a byte string literal"
}

#[derive(Copy, Clone)]
pub struct StringAdd;

impl LintPass for StringAdd {
    fn get_lints(&self) -> LintArray {
        lint_array!(STRING_ADD, STRING_ADD_ASSIGN)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for StringAdd {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        if let ExprBinary(Spanned { node: BiAdd, .. }, ref left, _) = e.node {
            if is_string(cx, left) {
                if let Allow = cx.current_level(STRING_ADD_ASSIGN) {
                    // the string_add_assign is allow, so no duplicates
                } else {
                    let parent = get_parent_expr(cx, e);
                    if let Some(p) = parent {
                        if let ExprAssign(ref target, _) = p.node {
                            // avoid duplicate matches
                            if SpanlessEq::new(cx).eq_expr(target, left) {
                                return;
                            }
                        }
                    }
                }
                span_lint(cx,
                          STRING_ADD,
                          e.span,
                          "you added something to a string. Consider using `String::push_str()` instead");
            }
        } else if let ExprAssign(ref target, ref src) = e.node {
            if is_string(cx, target) && is_add(cx, src, target) {
                span_lint(cx,
                          STRING_ADD_ASSIGN,
                          e.span,
                          "you assigned the result of adding something to this string. Consider using \
                           `String::push_str()` instead");
            }
        }
    }
}

fn is_string(cx: &LateContext, e: &Expr) -> bool {
    match_type(cx, walk_ptrs_ty(cx.tables.expr_ty(e)), &paths::STRING)
}

fn is_add(cx: &LateContext, src: &Expr, target: &Expr) -> bool {
    match src.node {
        ExprBinary(Spanned { node: BiAdd, .. }, ref left, _) => SpanlessEq::new(cx).eq_expr(target, left),
        ExprBlock(ref block) => {
            block.stmts.is_empty() && block.expr.as_ref().map_or(false, |expr| is_add(cx, expr, target))
        },
        _ => false,
    }
}

#[derive(Copy, Clone)]
pub struct StringLitAsBytes;

impl LintPass for StringLitAsBytes {
    fn get_lints(&self) -> LintArray {
        lint_array!(STRING_LIT_AS_BYTES)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for StringLitAsBytes {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        use std::ascii::AsciiExt;
        use syntax::ast::LitKind;
        use utils::{snippet, in_macro};

        if let ExprMethodCall(ref name, _, ref args) = e.node {
            if name.node == "as_bytes" {
                if let ExprLit(ref lit) = args[0].node {
                    if let LitKind::Str(ref lit_content, _) = lit.node {
                        if lit_content.as_str().chars().all(|c| c.is_ascii()) && !in_macro(args[0].span) {
                            span_lint_and_sugg(cx,
                                               STRING_LIT_AS_BYTES,
                                               e.span,
                                               "calling `as_bytes()` on a string literal",
                                               "consider using a byte string literal instead",
                                               format!("b{}", snippet(cx, args[0].span, r#""foo""#)));
                        }
                    }
                }
            }
        }
    }
}
