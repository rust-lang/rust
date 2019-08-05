//! Checks for continue statements in loops that are redundant.
//!
//! For example, the lint would catch
//!
//! ```rust
//! let mut a = 1;
//! let x = true;
//!
//! while a < 5 {
//!     a = 6;
//!     if x {
//!         // ...
//!     } else {
//!         continue;
//!     }
//!     println!("Hello, world");
//! }
//! ```
//!
//! And suggest something like this:
//!
//! ```rust
//! let mut a = 1;
//! let x = true;
//!
//! while a < 5 {
//!     a = 6;
//!     if x {
//!         // ...
//!         println!("Hello, world");
//!     }
//! }
//! ```
//!
//! This lint is **warn** by default.
use rustc::lint::{EarlyContext, EarlyLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use std::borrow::Cow;
use syntax::ast;
use syntax::source_map::{original_sp, DUMMY_SP};

use crate::utils::{in_macro_or_desugar, snippet, snippet_block, span_help_and_lint, trim_multiline};

declare_clippy_lint! {
    /// **What it does:** The lint checks for `if`-statements appearing in loops
    /// that contain a `continue` statement in either their main blocks or their
    /// `else`-blocks, when omitting the `else`-block possibly with some
    /// rearrangement of code can make the code easier to understand.
    ///
    /// **Why is this bad?** Having explicit `else` blocks for `if` statements
    /// containing `continue` in their THEN branch adds unnecessary branching and
    /// nesting to the code. Having an else block containing just `continue` can
    /// also be better written by grouping the statements following the whole `if`
    /// statement within the THEN block and omitting the else block completely.
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    /// ```rust
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
    /// ```rust
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
    /// ```rust
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
    /// ```rust
    /// # fn waiting() -> bool { false }
    /// loop {
    ///     if waiting() {
    ///         continue;
    ///     }
    ///     // Do something useful
    ///     # break;
    /// }
    /// ```
    pub NEEDLESS_CONTINUE,
    pedantic,
    "`continue` statements that can be replaced by a rearrangement of code"
}

declare_lint_pass!(NeedlessContinue => [NEEDLESS_CONTINUE]);

impl EarlyLintPass for NeedlessContinue {
    fn check_expr(&mut self, ctx: &EarlyContext<'_>, expr: &ast::Expr) {
        if !in_macro_or_desugar(expr.span) {
            check_and_warn(ctx, expr);
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
/// - The expression node is a block with the first statement being a
/// `continue`.
fn needless_continue_in_else(else_expr: &ast::Expr, label: Option<&ast::Label>) -> bool {
    match else_expr.node {
        ast::ExprKind::Block(ref else_block, _) => is_first_block_stmt_continue(else_block, label),
        ast::ExprKind::Continue(l) => compare_labels(label, l.as_ref()),
        _ => false,
    }
}

fn is_first_block_stmt_continue(block: &ast::Block, label: Option<&ast::Label>) -> bool {
    block.stmts.get(0).map_or(false, |stmt| match stmt.node {
        ast::StmtKind::Semi(ref e) | ast::StmtKind::Expr(ref e) => {
            if let ast::ExprKind::Continue(ref l) = e.node {
                compare_labels(label, l.as_ref())
            } else {
                false
            }
        },
        _ => false,
    })
}

/// If the `continue` has a label, check it matches the label of the loop.
fn compare_labels(loop_label: Option<&ast::Label>, continue_label: Option<&ast::Label>) -> bool {
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
    F: FnMut(&ast::Block, Option<&ast::Label>),
{
    if let ast::ExprKind::While(_, loop_block, label)
    | ast::ExprKind::ForLoop(_, _, loop_block, label)
    | ast::ExprKind::Loop(loop_block, label) = &expr.node
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
    F: FnMut(&ast::Expr, &ast::Expr, &ast::Block, &ast::Expr),
{
    match stmt.node {
        ast::StmtKind::Semi(ref e) | ast::StmtKind::Expr(ref e) => {
            if let ast::ExprKind::If(ref cond, ref if_block, Some(ref else_expr)) = e.node {
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
    if_block: &'a ast::Block,
    /// The `else` block of the `if` statement.
    /// Note that we only work with `if` exprs that have an `else` branch.
    else_expr: &'a ast::Expr,
    /// The 0-based index of the `if` statement in the containing loop block.
    stmt_idx: usize,
    /// The statements of the loop block.
    block_stmts: &'a [ast::Stmt],
}

const MSG_REDUNDANT_ELSE_BLOCK: &str = "This else block is redundant.\n";

const MSG_ELSE_BLOCK_NOT_NEEDED: &str = "There is no need for an explicit `else` block for this `if` \
                                         expression\n";

const DROP_ELSE_BLOCK_AND_MERGE_MSG: &str = "Consider dropping the else clause and merging the code that \
                                             follows (in the loop) with the if block, like so:\n";

const DROP_ELSE_BLOCK_MSG: &str = "Consider dropping the else clause, and moving out the code in the else \
                                   block, like so:\n";

fn emit_warning<'a>(ctx: &EarlyContext<'_>, data: &'a LintData<'_>, header: &str, typ: LintType) {
    // snip    is the whole *help* message that appears after the warning.
    // message is the warning message.
    // expr    is the expression which the lint warning message refers to.
    let (snip, message, expr) = match typ {
        LintType::ContinueInsideElseBlock => (
            suggestion_snippet_for_continue_inside_else(ctx, data, header),
            MSG_REDUNDANT_ELSE_BLOCK,
            data.else_expr,
        ),
        LintType::ContinueInsideThenBlock => (
            suggestion_snippet_for_continue_inside_if(ctx, data, header),
            MSG_ELSE_BLOCK_NOT_NEEDED,
            data.if_expr,
        ),
    };
    span_help_and_lint(ctx, NEEDLESS_CONTINUE, expr.span, message, &snip);
}

fn suggestion_snippet_for_continue_inside_if<'a>(
    ctx: &EarlyContext<'_>,
    data: &'a LintData<'_>,
    header: &str,
) -> String {
    let cond_code = snippet(ctx, data.if_cond.span, "..");

    let if_code = format!("if {} {{\n    continue;\n}}\n", cond_code);
    /* ^^^^--- Four spaces of indentation. */
    // region B
    let else_code = snippet(ctx, data.else_expr.span, "..").into_owned();
    let else_code = erode_block(&else_code);
    let else_code = trim_multiline(Cow::from(else_code), false);

    let mut ret = String::from(header);
    ret.push_str(&if_code);
    ret.push_str(&else_code);
    ret.push_str("\n...");
    ret
}

fn suggestion_snippet_for_continue_inside_else<'a>(
    ctx: &EarlyContext<'_>,
    data: &'a LintData<'_>,
    header: &str,
) -> String {
    let cond_code = snippet(ctx, data.if_cond.span, "..");
    let mut if_code = format!("if {} {{\n", cond_code);

    // Region B
    let block_code = &snippet(ctx, data.if_block.span, "..").into_owned();
    let block_code = erode_block(block_code);
    let block_code = trim_multiline(Cow::from(block_code), false);

    if_code.push_str(&block_code);

    // Region C
    // These is the code in the loop block that follows the if/else construction
    // we are complaining about. We want to pull all of this code into the
    // `then` block of the `if` statement.
    let to_annex = data.block_stmts[data.stmt_idx + 1..]
        .iter()
        .map(|stmt| original_sp(stmt.span, DUMMY_SP))
        .map(|span| snippet_block(ctx, span, "..").into_owned())
        .collect::<Vec<_>>()
        .join("\n");

    let mut ret = String::from(header);

    ret.push_str(&if_code);
    ret.push_str("\n// Merged code follows...");
    ret.push_str(&to_annex);
    ret.push_str("\n}\n");
    ret
}

fn check_and_warn<'a>(ctx: &EarlyContext<'_>, expr: &'a ast::Expr) {
    with_loop_block(expr, |loop_block, label| {
        for (i, stmt) in loop_block.stmts.iter().enumerate() {
            with_if_expr(stmt, |if_expr, cond, then_block, else_expr| {
                let data = &LintData {
                    stmt_idx: i,
                    if_expr,
                    if_cond: cond,
                    if_block: then_block,
                    else_expr,
                    block_stmts: &loop_block.stmts,
                };
                if needless_continue_in_else(else_expr, label) {
                    emit_warning(
                        ctx,
                        data,
                        DROP_ELSE_BLOCK_AND_MERGE_MSG,
                        LintType::ContinueInsideElseBlock,
                    );
                } else if is_first_block_stmt_continue(then_block, label) {
                    emit_warning(ctx, data, DROP_ELSE_BLOCK_MSG, LintType::ContinueInsideThenBlock);
                }
            });
        }
    });
}

/// Eats at `s` from the end till a closing brace `}` is encountered, and then
/// continues eating till a non-whitespace character is found.
/// e.g., the string
///
/// ```rust
/// {
///     let x = 5;
/// }
/// ```
///
/// is transformed to
///
/// ```ignore
///     {
///         let x = 5;
/// ```
///
/// NOTE: when there is no closing brace in `s`, `s` is _not_ preserved, i.e.,
/// an empty string will be returned in that case.
pub fn erode_from_back(s: &str) -> String {
    let mut ret = String::from(s);
    while ret.pop().map_or(false, |c| c != '}') {}
    while let Some(c) = ret.pop() {
        if !c.is_whitespace() {
            ret.push(c);
            break;
        }
    }
    ret
}

/// Eats at `s` from the front by first skipping all leading whitespace. Then,
/// any number of opening braces are eaten, followed by any number of newlines.
/// e.g.,  the string
///
/// ```ignore
///         {
///             something();
///             inside_a_block();
///         }
/// ```
///
/// is transformed to
///
/// ```ignore
///             something();
///             inside_a_block();
///         }
/// ```
pub fn erode_from_front(s: &str) -> String {
    s.chars()
        .skip_while(|c| c.is_whitespace())
        .skip_while(|c| *c == '{')
        .skip_while(|c| *c == '\n')
        .collect::<String>()
}

/// If `s` contains the code for a block, delimited by braces, this function
/// tries to get the contents of the block. If there is no closing brace
/// present,
/// an empty string is returned.
pub fn erode_block(s: &str) -> String {
    erode_from_back(&erode_from_front(s))
}
