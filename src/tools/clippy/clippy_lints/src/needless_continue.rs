use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::source::{indent_of, snippet, snippet_block};
use rustc_ast::{Block, Label, ast};
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// The lint checks for `if`-statements appearing in loops
    /// that contain a `continue` statement in either their main blocks or their
    /// `else`-blocks, when omitting the `else`-block possibly with some
    /// rearrangement of code can make the code easier to understand.
    /// The lint also checks if the last statement in the loop is a `continue`
    ///
    /// ### Why is this bad?
    /// Having explicit `else` blocks for `if` statements
    /// containing `continue` in their THEN branch adds unnecessary branching and
    /// nesting to the code. Having an else block containing just `continue` can
    /// also be better written by grouping the statements following the whole `if`
    /// statement within the THEN block and omitting the else block completely.
    ///
    /// ### Example
    /// ```no_run
    /// # fn condition() -> bool { false }
    /// # fn update_condition() {}
    /// # let x = false;
    /// while condition() {
    ///     update_condition();
    ///     if x {
    ///         // ...
    ///     } else {
    ///         continue;
    ///     }
    ///     println!("Hello, world");
    /// }
    /// ```
    ///
    /// Could be rewritten as
    ///
    /// ```no_run
    /// # fn condition() -> bool { false }
    /// # fn update_condition() {}
    /// # let x = false;
    /// while condition() {
    ///     update_condition();
    ///     if x {
    ///         // ...
    ///         println!("Hello, world");
    ///     }
    /// }
    /// ```
    ///
    /// As another example, the following code
    ///
    /// ```no_run
    /// # fn waiting() -> bool { false }
    /// loop {
    ///     if waiting() {
    ///         continue;
    ///     } else {
    ///         // Do something useful
    ///     }
    ///     # break;
    /// }
    /// ```
    /// Could be rewritten as
    ///
    /// ```no_run
    /// # fn waiting() -> bool { false }
    /// loop {
    ///     if waiting() {
    ///         continue;
    ///     }
    ///     // Do something useful
    ///     # break;
    /// }
    /// ```
    ///
    /// ```rust
    /// # use std::io::ErrorKind;
    ///
    /// fn foo() -> ErrorKind { ErrorKind::NotFound }
    /// for _ in 0..10 {
    ///     match foo() {
    ///         ErrorKind::NotFound => {
    ///             eprintln!("not found");
    ///             continue
    ///         }
    ///         ErrorKind::TimedOut => {
    ///             eprintln!("timeout");
    ///             continue
    ///         }
    ///         _ => {
    ///             eprintln!("other error");
    ///             continue
    ///         }
    ///     }
    /// }
    /// ```
    /// Could be rewritten as
    ///
    ///
    /// ```rust
    /// # use std::io::ErrorKind;
    ///
    /// fn foo() -> ErrorKind { ErrorKind::NotFound }
    /// for _ in 0..10 {
    ///     match foo() {
    ///         ErrorKind::NotFound => {
    ///             eprintln!("not found");
    ///         }
    ///         ErrorKind::TimedOut => {
    ///             eprintln!("timeout");
    ///         }
    ///         _ => {
    ///             eprintln!("other error");
    ///         }
    ///     }
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub NEEDLESS_CONTINUE,
    pedantic,
    "`continue` statements that can be replaced by a rearrangement of code"
}

declare_lint_pass!(NeedlessContinue => [NEEDLESS_CONTINUE]);

impl EarlyLintPass for NeedlessContinue {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &ast::Expr) {
        if !expr.span.from_expansion() {
            check_and_warn(cx, expr);
        }
    }
}

/* This lint has to mainly deal with two cases of needless continue
 * statements. */
// Case 1 [Continue inside else block]:
//
//     loop {
//         // region A
//         if cond {
//             // region B
//         } else {
//             continue;
//         }
//         // region C
//     }
//
// This code can better be written as follows:
//
//     loop {
//         // region A
//         if cond {
//             // region B
//             // region C
//         }
//     }
//
// Case 2 [Continue inside then block]:
//
//     loop {
//       // region A
//       if cond {
//           continue;
//           // potentially more code here.
//       } else {
//           // region B
//       }
//       // region C
//     }
//
//
// This snippet can be refactored to:
//
//     loop {
//       // region A
//       if !cond {
//           // region B
//           // region C
//       }
//     }
//

/// Given an expression, returns true if either of the following is true
///
/// - The expression is a `continue` node.
/// - The expression node is a block with the first statement being a `continue`.
fn needless_continue_in_else(else_expr: &ast::Expr, label: Option<&Label>) -> bool {
    match else_expr.kind {
        ast::ExprKind::Block(ref else_block, _) => is_first_block_stmt_continue(else_block, label),
        ast::ExprKind::Continue(l) => compare_labels(label, l.as_ref()),
        _ => false,
    }
}

fn is_first_block_stmt_continue(block: &Block, label: Option<&Label>) -> bool {
    block.stmts.first().is_some_and(|stmt| match stmt.kind {
        ast::StmtKind::Semi(ref e) | ast::StmtKind::Expr(ref e) => {
            if let ast::ExprKind::Continue(ref l) = e.kind {
                compare_labels(label, l.as_ref())
            } else {
                false
            }
        },
        _ => false,
    })
}

/// If the `continue` has a label, check it matches the label of the loop.
fn compare_labels(loop_label: Option<&Label>, continue_label: Option<&Label>) -> bool {
    match (loop_label, continue_label) {
        // `loop { continue; }` or `'a loop { continue; }`
        (_, None) => true,
        // `loop { continue 'a; }`
        (None, _) => false,
        // `'a loop { continue 'a; }` or `'a loop { continue 'b; }`
        (Some(x), Some(y)) => x.ident == y.ident,
    }
}

/// If `expr` is a loop expression (while/while let/for/loop), calls `func` with
/// the AST object representing the loop block of `expr`.
fn with_loop_block<F>(expr: &ast::Expr, mut func: F)
where
    F: FnMut(&Block, Option<&Label>),
{
    if let ast::ExprKind::While(_, loop_block, label)
    | ast::ExprKind::ForLoop {
        body: loop_block,
        label,
        ..
    }
    | ast::ExprKind::Loop(loop_block, label, ..) = &expr.kind
    {
        func(loop_block, label.as_ref());
    }
}

/// If `stmt` is an if expression node with an `else` branch, calls func with
/// the
/// following:
///
/// - The `if` expression itself,
/// - The `if` condition expression,
/// - The `then` block, and
/// - The `else` expression.
fn with_if_expr<F>(stmt: &ast::Stmt, mut func: F)
where
    F: FnMut(&ast::Expr, &ast::Expr, &Block, &ast::Expr),
{
    match stmt.kind {
        ast::StmtKind::Semi(ref e) | ast::StmtKind::Expr(ref e) => {
            if let ast::ExprKind::If(ref cond, ref if_block, Some(ref else_expr)) = e.kind {
                func(e, cond, if_block, else_expr);
            }
        },
        _ => {},
    }
}

/// A type to distinguish between the two distinct cases this lint handles.
#[derive(Copy, Clone, Debug)]
enum LintType {
    ContinueInsideElseBlock,
    ContinueInsideThenBlock,
}

/// Data we pass around for construction of help messages.
struct LintData<'a> {
    /// The `if` expression encountered in the above loop.
    if_expr: &'a ast::Expr,
    /// The condition expression for the above `if`.
    if_cond: &'a ast::Expr,
    /// The `then` block of the `if` statement.
    if_block: &'a Block,
    /// The `else` block of the `if` statement.
    /// Note that we only work with `if` exprs that have an `else` branch.
    else_expr: &'a ast::Expr,
    /// The 0-based index of the `if` statement in the containing loop block.
    stmt_idx: usize,
    /// The statements of the loop block.
    loop_block: &'a Block,
}

const MSG_REDUNDANT_CONTINUE_EXPRESSION: &str = "this `continue` expression is redundant";

const MSG_REDUNDANT_ELSE_BLOCK: &str = "this `else` block is redundant";

const MSG_ELSE_BLOCK_NOT_NEEDED: &str = "there is no need for an explicit `else` block for this `if` \
                                         expression";

const DROP_ELSE_BLOCK_AND_MERGE_MSG: &str = "consider dropping the `else` clause and merging the code that \
                                             follows (in the loop) with the `if` block";

const DROP_ELSE_BLOCK_MSG: &str = "consider dropping the `else` clause";

const DROP_CONTINUE_EXPRESSION_MSG: &str = "consider dropping the `continue` expression";

fn emit_warning(cx: &EarlyContext<'_>, data: &LintData<'_>, header: &str, typ: LintType) {
    // snip    is the whole *help* message that appears after the warning.
    // message is the warning message.
    // expr    is the expression which the lint warning message refers to.
    let (snip, message, expr) = match typ {
        LintType::ContinueInsideElseBlock => (
            suggestion_snippet_for_continue_inside_else(cx, data),
            MSG_REDUNDANT_ELSE_BLOCK,
            data.else_expr,
        ),
        LintType::ContinueInsideThenBlock => (
            suggestion_snippet_for_continue_inside_if(cx, data),
            MSG_ELSE_BLOCK_NOT_NEEDED,
            data.if_expr,
        ),
    };
    span_lint_and_help(
        cx,
        NEEDLESS_CONTINUE,
        expr.span,
        message,
        None,
        format!("{header}\n{snip}"),
    );
}

fn suggestion_snippet_for_continue_inside_if(cx: &EarlyContext<'_>, data: &LintData<'_>) -> String {
    let cond_code = snippet(cx, data.if_cond.span, "..");

    let continue_code = snippet_block(cx, data.if_block.span, "..", Some(data.if_expr.span));

    let else_code = snippet_block(cx, data.else_expr.span, "..", Some(data.if_expr.span));

    let indent_if = indent_of(cx, data.if_expr.span).unwrap_or(0);
    format!(
        "{indent}if {cond_code} {continue_code}\n{indent}{else_code}",
        indent = " ".repeat(indent_if),
    )
}

fn suggestion_snippet_for_continue_inside_else(cx: &EarlyContext<'_>, data: &LintData<'_>) -> String {
    let cond_code = snippet(cx, data.if_cond.span, "..");

    // Region B
    let block_code = erode_from_back(&snippet_block(cx, data.if_block.span, "..", Some(data.if_expr.span)));

    // Region C
    // These is the code in the loop block that follows the if/else construction
    // we are complaining about. We want to pull all of this code into the
    // `then` block of the `if` statement.
    let indent = span_of_first_expr_in_block(data.if_block)
        .and_then(|span| indent_of(cx, span))
        .unwrap_or(0);
    let to_annex = data.loop_block.stmts[data.stmt_idx + 1..]
        .iter()
        .map(|stmt| {
            let span = cx.sess().source_map().stmt_span(stmt.span, data.loop_block.span);
            let snip = snippet_block(cx, span, "..", None);
            snip.lines()
                .map(|line| format!("{}{line}", " ".repeat(indent)))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let indent_if = indent_of(cx, data.if_expr.span).unwrap_or(0);
    format!(
        "{indent_if}if {cond_code} {block_code}\n{indent}// merged code follows:\n{to_annex}\n{indent_if}}}",
        indent = " ".repeat(indent),
        indent_if = " ".repeat(indent_if),
    )
}

fn check_last_stmt_in_expr<F>(inner_expr: &ast::Expr, func: &F)
where
    F: Fn(Option<&Label>, Span),
{
    match &inner_expr.kind {
        ast::ExprKind::Continue(continue_label) => {
            func(continue_label.as_ref(), inner_expr.span);
        },
        ast::ExprKind::If(_, then_block, else_block) => {
            check_last_stmt_in_block(then_block, func);
            if let Some(else_block) = else_block {
                check_last_stmt_in_expr(else_block, func);
            }
        },
        ast::ExprKind::Match(_, arms, _) => {
            for arm in arms {
                if let Some(expr) = &arm.body {
                    check_last_stmt_in_expr(expr, func);
                }
            }
        },
        ast::ExprKind::Block(b, _) => {
            check_last_stmt_in_block(b, func);
        },
        _ => {},
    }
}

fn check_last_stmt_in_block<F>(b: &Block, func: &F)
where
    F: Fn(Option<&Label>, Span),
{
    if let Some(last_stmt) = b.stmts.last()
        && let ast::StmtKind::Expr(inner_expr) | ast::StmtKind::Semi(inner_expr) = &last_stmt.kind
    {
        check_last_stmt_in_expr(inner_expr, func);
    }
}

fn check_and_warn(cx: &EarlyContext<'_>, expr: &ast::Expr) {
    with_loop_block(expr, |loop_block, label| {
        let p = |continue_label: Option<&Label>, span: Span| {
            if compare_labels(label, continue_label) {
                span_lint_and_help(
                    cx,
                    NEEDLESS_CONTINUE,
                    span,
                    MSG_REDUNDANT_CONTINUE_EXPRESSION,
                    None,
                    DROP_CONTINUE_EXPRESSION_MSG,
                );
            }
        };

        let stmts = &loop_block.stmts;
        for (i, stmt) in stmts.iter().enumerate() {
            let mut maybe_emitted_in_if = false;
            with_if_expr(stmt, |if_expr, cond, then_block, else_expr| {
                let data = &LintData {
                    if_expr,
                    if_cond: cond,
                    if_block: then_block,
                    else_expr,
                    stmt_idx: i,
                    loop_block,
                };

                maybe_emitted_in_if = true;
                if needless_continue_in_else(else_expr, label) {
                    emit_warning(
                        cx,
                        data,
                        DROP_ELSE_BLOCK_AND_MERGE_MSG,
                        LintType::ContinueInsideElseBlock,
                    );
                } else if is_first_block_stmt_continue(then_block, label) {
                    emit_warning(cx, data, DROP_ELSE_BLOCK_MSG, LintType::ContinueInsideThenBlock);
                } else {
                    maybe_emitted_in_if = false;
                }
            });

            if i == stmts.len() - 1 && !maybe_emitted_in_if {
                check_last_stmt_in_block(loop_block, &p);
            }
        }
    });
}

/// Eats at `s` from the end till a closing brace `}` is encountered, and then continues eating
/// till a non-whitespace character is found.  e.g., the string. If no closing `}` is present, the
/// string will be preserved.
///
/// ```no_run
/// {
///     let x = 5;
/// }
/// ```
///
/// is transformed to
///
/// ```text
///     {
///         let x = 5;
/// ```
#[must_use]
fn erode_from_back(s: &str) -> String {
    let mut ret = s.to_string();
    while ret.pop().is_some_and(|c| c != '}') {}
    while let Some(c) = ret.pop() {
        if !c.is_whitespace() {
            ret.push(c);
            break;
        }
    }
    if ret.is_empty() { s.to_string() } else { ret }
}

fn span_of_first_expr_in_block(block: &Block) -> Option<Span> {
    block.stmts.first().map(|stmt| stmt.span)
}

#[cfg(test)]
mod test {
    use super::erode_from_back;

    #[test]
    #[rustfmt::skip]
    fn test_erode_from_back() {
        let input = "\
{
    let x = 5;
    let y = format!(\"{}\", 42);
}";

        let expected = "\
{
    let x = 5;
    let y = format!(\"{}\", 42);";

        let got = erode_from_back(input);
        assert_eq!(expected, got);
    }

    #[test]
    #[rustfmt::skip]
    fn test_erode_from_back_no_brace() {
        let input = "\
let x = 5;
let y = something();
";
        let expected = input;
        let got = erode_from_back(input);
        assert_eq!(expected, got);
    }
}
