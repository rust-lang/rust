use rustc::hir::*;
use rustc::lint::*;
use syntax::codemap::Spanned;
use utils::SpanlessEq;
use utils::{match_type, paths, span_lint, span_lint_and_then, walk_ptrs_ty, get_parent_expr};

/// **What it does:** This lint matches code of the form `x = x + y` (without `let`!).
///
/// **Why is this bad?** It's not really bad, but some people think that the `.push_str(_)` method
/// is more readable.
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
    "using `x = x + ..` where x is a `String`; suggests using `push_str()` instead"
}

/// **What it does:** The `string_add` lint matches all instances of `x + _` where `x` is of type
/// `String`, but only if [`string_add_assign`](#string_add_assign) does *not* match.
///
/// **Why is this bad?** It's not bad in and of itself. However, this particular `Add`
/// implementation is asymmetric (the other operand need not be `String`, but `x` does), while
/// addition as mathematically defined is symmetric, also the `String::push_str(_)` function is a
/// perfectly good replacement. Therefore some dislike it and wish not to have it in their code.
///
/// That said, other people think that string addition, having a long tradition in other languages
/// is actually fine, which is why we decided to make this particular lint `allow` by default.
///
/// **Known problems:** None
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
    "using `x + ..` where x is a `String`; suggests using `push_str()` instead"
}

/// **What it does:** This lint matches the `as_bytes` method called on string
/// literals that contain only ASCII characters.
///
/// **Why is this bad?** Byte string literals (e.g. `b"foo"`) can be used instead. They are shorter
/// but less discoverable than `as_bytes()`.
///
/// **Example:**
///
/// ```rust
/// let bs = "a byte string".as_bytes();
/// ```
declare_lint! {
    pub STRING_LIT_AS_BYTES,
    Warn,
    "calling `as_bytes` on a string literal; suggests using a byte string literal instead"
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
        if let ExprBinary(Spanned { node: BiAdd, .. }, ref left, _) = e.node {
            if is_string(cx, left) {
                if let Allow = cx.current_level(STRING_ADD_ASSIGN) {
                    // the string_add_assign is allow, so no duplicates
                } else {
                    let parent = get_parent_expr(cx, e);
                    if let Some(ref p) = parent {
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
    match_type(cx, walk_ptrs_ty(cx.tcx.expr_ty(e)), &paths::STRING)
}

fn is_add(cx: &LateContext, src: &Expr, target: &Expr) -> bool {
    match src.node {
        ExprBinary(Spanned { node: BiAdd, .. }, ref left, _) => SpanlessEq::new(cx).eq_expr(target, left),
        ExprBlock(ref block) => {
            block.stmts.is_empty() && block.expr.as_ref().map_or(false, |expr| is_add(cx, expr, target))
        }
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

impl LateLintPass for StringLitAsBytes {
    fn check_expr(&mut self, cx: &LateContext, e: &Expr) {
        use std::ascii::AsciiExt;
        use syntax::ast::LitKind;
        use utils::{snippet, in_macro};

        if let ExprMethodCall(ref name, _, ref args) = e.node {
            if name.node.as_str() == "as_bytes" {
                if let ExprLit(ref lit) = args[0].node {
                    if let LitKind::Str(ref lit_content, _) = lit.node {
                        if lit_content.chars().all(|c| c.is_ascii()) && !in_macro(cx, args[0].span) {
                            span_lint_and_then(cx,
                                               STRING_LIT_AS_BYTES,
                                               e.span,
                                               "calling `as_bytes()` on a string literal",
                                               |db| {
                                                   let sugg = format!("b{}", snippet(cx, args[0].span, r#""foo""#));
                                                   db.span_suggestion(e.span,
                                                                      "consider using a byte string literal instead",
                                                                      sugg);
                                               });

                        }
                    }
                }
            }
        }
    }
}
