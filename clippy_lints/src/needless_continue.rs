//! Checks for continue statements in loops that are redundant.
//!
//! For example, the lint would catch
//!
//! ```
//! while condition() {
//!     update_condition();
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
//! ```
//! while condition() {
//!     update_condition();
//!     if x {
//!         // ...
//!         println!("Hello, world");
//!     }
//! }
//! ```
//!
//! This lint is **warn** by default.
use std;
use rustc::lint::*;
use syntax::ast;
use syntax::codemap::{original_sp,DUMMY_SP};

use utils::{in_macro, span_help_and_lint, snippet_block, snippet};
use self::LintType::*;

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
/// loop {
///     if waiting() {
///         continue;
///     } else {
///         // Do something useful
///     }
/// }
/// ```
/// Could be rewritten as
///
/// ```rust
/// loop {
///     if waiting() {
///         continue;
///     }
///     // Do something useful
/// }
/// ```
declare_lint! {
    pub NEEDLESS_CONTINUE,
    Warn,
    "`continue` statements that can be replaced by a rearrangement of code"
}

#[derive(Copy,Clone)]
pub struct NeedlessContinue;

impl LintPass for NeedlessContinue {
    fn get_lints(&self) -> LintArray {
        lint_array!(NEEDLESS_CONTINUE)
    }
}

impl EarlyLintPass for NeedlessContinue {
    fn check_expr(&mut self, ctx: &EarlyContext, expr: &ast::Expr) {
        if !in_macro(ctx, expr.span) {
            check_and_warn(ctx, expr);
        }
    }
}

/// Given an Expr, returns true if either of the following is true
///
/// - The Expr is a `continue` node.
/// - The Expr node is a block with the first statement being a `continue`.
///
fn needless_continue_in_else(else_expr: &ast::Expr) -> bool {
    let mut found = false;
    match else_expr.node {
        ast::ExprKind::Block(ref else_block) => {
            found = is_first_block_stmt_continue(else_block);
        },
        ast::ExprKind::Continue(_) => { found = true },
        _ => { },
    };
    found
}

fn is_first_block_stmt_continue(block: &ast::Block) -> bool {
    let mut ret = false;
    block.stmts.get(0).map(|stmt| {
        if_let_chain! {[
            let ast::StmtKind::Semi(ref e) = stmt.node,
            let ast::ExprKind::Continue(_) = e.node,
        ], {
            ret = true;
        }}
    });
    ret
}

/// If `expr` is a loop expression (while/while let/for/loop), calls `func` with
/// the AST object representing the loop block of `expr`.
fn with_loop_block<F>(expr: &ast::Expr, mut func: F) where F: FnMut(&ast::Block) {
    match expr.node {
        ast::ExprKind::While(_, ref loop_block, _)       |
        ast::ExprKind::WhileLet(_, _, ref loop_block, _) |
        ast::ExprKind::ForLoop( _, _, ref loop_block, _) |
        ast::ExprKind::Loop(ref loop_block, _)           => func(loop_block),
        _ => {},
    }
}

/// If `stmt` is an if expression node with an else branch, calls func with the
/// following:
///
/// - The if Expr,
/// - The if condition Expr,
/// - The then block of this if Expr, and
/// - The else expr.
///
fn with_if_expr<F>(stmt: &ast::Stmt, mut func: F)
        where F: FnMut(&ast::Expr, &ast::Expr, &ast::Block, &ast::Expr) {
    match stmt.node {
        ast::StmtKind::Semi(ref e) |
        ast::StmtKind::Expr(ref e) => {
            if let ast::ExprKind::If(ref cond, ref if_block, Some(ref else_expr)) = e.node {
                func(e, cond, if_block, else_expr);
            }
        },
        _ => { },
    }
}

/// A type to distinguish between the two distinct cases this lint handles.
enum LintType {
    ContinueInsideElseBlock,
    ContinueInsideThenBlock,
}

/// Data we pass around for construction of help messages.
struct LintData<'a> {
    if_expr:     &'a ast::Expr,    // The `if` expr encountered in the above loop.
    if_cond:     &'a ast::Expr,    // The condition expression for the above `if`.
    if_block:    &'a ast::Block,   // The `then` block of the `if` statement.
    else_expr:   &'a ast::Expr,    /* The `else` block of the `if` statement.
                                      Note that we only work with `if` exprs that
                                      have an `else` branch. */
    stmt_idx:    usize,            /* The 0-based index of the `if` statement in
                                      the containing loop block. */
    block_stmts: &'a [ast::Stmt],  // The statements of the loop block.
}

const MSG_REDUNDANT_ELSE_BLOCK: &'static str = "This else block is redundant.\n";

const MSG_ELSE_BLOCK_NOT_NEEDED: &'static str = "There is no need for an explicit `else` block for this `if` expression\n";

const DROP_ELSE_BLOCK_AND_MERGE_MSG: &'static str =
    "Consider dropping the else clause and merging the code that follows (in the loop) with the if block, like so:\n";

const DROP_ELSE_BLOCK_MSG: &'static str =
    "Consider dropping the else clause, and moving out the code in the else block, like so:\n";


fn emit_warning<'a>(ctx: &EarlyContext,
                    data: &'a LintData,
                    header: &str,
                    typ: LintType) {

    // snip    is the whole *help* message that appears after the warning.
    // message is the warning message.
    // expr    is the expression which the lint warning message refers to.
    let (snip, message, expr) = match typ {
        ContinueInsideElseBlock => {
            (suggestion_snippet_for_continue_inside_else(ctx, data, header),
             MSG_REDUNDANT_ELSE_BLOCK,
             data.else_expr)
        },
        ContinueInsideThenBlock => {
            (suggestion_snippet_for_continue_inside_if(ctx, data, header),
             MSG_ELSE_BLOCK_NOT_NEEDED,
             data.if_expr)
        }
    };
    span_help_and_lint(ctx, NEEDLESS_CONTINUE, expr.span, message, &snip);
}

fn suggestion_snippet_for_continue_inside_if<'a>(ctx: &EarlyContext,
                                                data: &'a LintData,
                                                header: &str) -> String {
    let cond_code = &snippet(ctx, data.if_cond.span, "..").into_owned();

    let if_code   = &format!("if {} {{\n    continue;\n}}\n", cond_code);
                                    /*  ^^^^--- Four spaces of indentation. */
    // region B
    let else_code = snippet(ctx, data.else_expr.span, "..").into_owned();
    let else_code = erode_block(&else_code);
    let else_code = trim_indent(&else_code, false);

    let mut ret = String::from(header);
    ret.push_str(&if_code);
    ret.push_str(&else_code);
    ret.push_str("\n...");
    ret
}

fn suggestion_snippet_for_continue_inside_else<'a>(ctx: &EarlyContext,
                                                   data: &'a LintData,
                                                   header: &str) -> String
{
    let cond_code = &snippet(ctx, data.if_cond.span, "..").into_owned();
    let mut if_code   = format!("if {} {{\n", cond_code);

    // Region B
    let block_code = &snippet(ctx, data.if_block.span, "..").into_owned();
    let block_code = erode_block(block_code);
    let block_code = trim_indent(&block_code, false);
    let block_code = left_pad_lines_with_spaces(&block_code, 4usize);

    if_code.push_str(&block_code);

    // Region C
    // These is the code in the loop block that follows the if/else construction
    // we are complaining about. We want to pull all of this code into the
    // `then` block of the `if` statement.
    let to_annex = data.block_stmts[data.stmt_idx+1..]
                   .iter()
                   .map(|stmt| {
                        original_sp(ctx.sess().codemap(), stmt.span, DUMMY_SP)
                    })
                   .map(|span| snippet_block(ctx, span, "..").into_owned())
                   .collect::<Vec<_>>().join("\n");

    let mut ret = String::from(header);
    ret.push_str(&align_snippets(&[&if_code,
                                   "\n// Merged code follows...",
                                   &to_annex]));
    ret.push_str("\n}\n");
    ret
}

fn check_and_warn<'a>(ctx: &EarlyContext, expr: &'a ast::Expr) {
    with_loop_block(expr, |loop_block| {
        for (i, stmt) in loop_block.stmts.iter().enumerate() {
            with_if_expr(stmt, |if_expr, cond, then_block, else_expr| {
                let data = &LintData {
                    stmt_idx:    i,
                    if_expr:     if_expr,
                    if_cond:     cond,
                    if_block:    then_block,
                    else_expr:   else_expr,
                    block_stmts: &loop_block.stmts,
                };
                if needless_continue_in_else(else_expr) {
                    emit_warning(ctx, data, DROP_ELSE_BLOCK_AND_MERGE_MSG, ContinueInsideElseBlock);
                } else if is_first_block_stmt_continue(then_block) {
                    emit_warning(ctx, data, DROP_ELSE_BLOCK_MSG, ContinueInsideThenBlock);
                }
            });
        }
    });
}

/// Eats at `s` from the end till a closing brace `}` is encountered, and then
/// continues eating till a non-whitespace character is found.
/// e.g., the string
///
/// "
/// {
///     let x = 5;
/// }
/// "
///
/// is transformed to
///
/// "
/// {
///     let x = 5;"
///
fn erode_from_back(s: &str) -> String {
    let mut ret = String::from(s);
    while ret.pop().map_or(false, |c| c != '}') { }
    while let Some(c) = ret.pop() {
        if !c.is_whitespace() {
            ret.push(c);
            break;
        }
    }
    ret
}

fn erode_from_front(s: &str) -> String {
    s.chars()
     .skip_while(|c| c.is_whitespace())
     .skip_while(|c| *c == '{')
     .skip_while(|c| *c == '\n')
     .collect::<String>()
}

fn erode_block(s: &str) -> String {
    erode_from_back(&erode_from_front(s))
}

fn is_all_whitespace(s: &str) -> bool { s.chars().all(|c| c.is_whitespace()) }

/// Returns true if a string is empty or just spaces.
fn is_null(s: &str) -> bool { s.is_empty() || is_all_whitespace(s) }

/// Returns the indentation level of a string. It just returns the count of
/// whitespace characters in the string before a non-whitespace character
/// is encountered.
fn indent_level(s: &str) -> usize {
    s.chars()
     .enumerate()
     .find(|&(_, c)| !c.is_whitespace())
     .map_or(0usize, |(i, _)| i)
}

/// Trims indentation from a snippet such that the line with the minimum
/// indentation has no indentation after the trasformation.
fn trim_indent(s: &str, skip_first_line: bool) -> String {
    let min_indent_level = s.lines()
                            .filter(|line| !is_null(line))
                            .skip(skip_first_line as usize)
                            .map(indent_level)
                            .min()
                            .unwrap_or(0usize);
    let ret = s.lines().map(|line| {
        if is_null(line) {
            String::from(line)
        } else {
            line.chars()
                .enumerate()
                .skip_while(|&(i, c)| c.is_whitespace() && i < min_indent_level)
                .map(|pair| pair.1)
                .collect::<String>()
        }
    }).collect::<Vec<String>>();
    ret.join("\n")
}

/// Add `n` spaces to the left of `s`.
fn left_pad_with_spaces(s: &str, n: usize) -> String {
    let mut new_s = std::iter::repeat(' '/* <-space */).take(n).collect::<String>();
    new_s.push_str(s);
    new_s
}

/// Add `n` spaces to the left of each line in `s` and return the result
/// in a new String.
fn left_pad_lines_with_spaces(s: &str, n: usize) -> String {
    s.lines()
     .map(|line| left_pad_with_spaces(line, n))
     .collect::<Vec<_>>()
     .join("\n")
}

/// Remove upto `n` whitespace characters from the beginning of `s`.
fn remove_whitespace_from_left(s: &str, n: usize) -> String {
    s.chars()
     .enumerate()
     .skip_while(|&(i, c)| i < n && c.is_whitespace())
     .map(|(_, c)| c)
     .collect::<String>()
}

/// Aligns two snippets such that the indentation level of the last non-empty,
/// non-space line of the first snippet matches the first non-empty, non-space
/// line of the second.
fn align_two_snippets(s: &str, t: &str) -> String {
    // indent level of the last nonempty, non-whitespace line of s.
    let target_ilevel = s.lines()
                         .rev()
                         .skip_while(|line| line.is_empty() || is_all_whitespace(line))
                         .next()
                         .map_or(0usize, indent_level);

    // We want to align the first nonempty, non-all-whitespace line of t to
    // have the same indent level as target_ilevel
    let level = t.lines()
                 .skip_while(|line| line.is_empty() || is_all_whitespace(line))
                 .next()
                 .map_or(0usize, indent_level);

    let add_or_not_remove = target_ilevel > level; /* when true, we add spaces,
                                                      otherwise eat. */

    let delta = if add_or_not_remove {
        target_ilevel - level
    } else {
        level - target_ilevel
    };

    let new_t = t.lines()
                 .filter(|line| !is_null(line))
                 .map(|line| if add_or_not_remove {
                     left_pad_with_spaces(line, delta)
                 } else {
                     remove_whitespace_from_left(line, delta)
                 })
                 .collect::<Vec<_>>().join("\n");

    format!("{}\n{}", s, new_t)
}

fn align_snippets(xs: &[&str]) -> String {
    match xs.len() {
        0 => String::from(""),
        _ => {
            let mut ret = String::new();
            ret.push_str(xs[0]);
            for x in xs.iter().skip(1usize) {
                ret = align_two_snippets(&ret, x);
            }
            ret
        }
    }
}

