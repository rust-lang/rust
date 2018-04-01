use std::ops::Deref;
use rustc::hir::*;
use rustc::hir::map::Node::{NodeImplItem, NodeItem};
use rustc::lint::*;
use syntax::ast::LitKind;
use syntax::symbol::InternedString;
use syntax_pos::Span;
use utils::{is_expn_of, match_def_path, match_path, resolve_node, span_lint, span_lint_and_sugg};
use utils::{opt_def_id, paths};

/// **What it does:** This lint warns when you use `println!("")` to
/// print a newline.
///
/// **Why is this bad?** You should use `println!()`, which is simpler.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// println!("");
/// ```
declare_clippy_lint! {
    pub PRINTLN_EMPTY_STRING,
    style,
    "using `println!(\"\")` with an empty string"
}

/// **What it does:** This lint warns when you use `print!()` with a format
/// string that
/// ends in a newline.
///
/// **Why is this bad?** You should use `println!()` instead, which appends the
/// newline.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// print!("Hello {}!\n", name);
/// ```
declare_clippy_lint! {
    pub PRINT_WITH_NEWLINE,
    style,
    "using `print!()` with a format string that ends in a newline"
}

/// **What it does:** Checks for printing on *stdout*. The purpose of this lint
/// is to catch debugging remnants.
///
/// **Why is this bad?** People often print on *stdout* while debugging an
/// application and might forget to remove those prints afterward.
///
/// **Known problems:** Only catches `print!` and `println!` calls.
///
/// **Example:**
/// ```rust
/// println!("Hello world!");
/// ```
declare_clippy_lint! {
    pub PRINT_STDOUT,
    restriction,
    "printing on stdout"
}

/// **What it does:** Checks for use of `Debug` formatting. The purpose of this
/// lint is to catch debugging remnants.
///
/// **Why is this bad?** The purpose of the `Debug` trait is to facilitate
/// debugging Rust code. It should not be used in in user-facing output.
///
/// **Example:**
/// ```rust
/// println!("{:?}", foo);
/// ```
declare_clippy_lint! {
    pub USE_DEBUG,
    restriction,
    "use of `Debug`-based formatting"
}

/// **What it does:** This lint warns about the use of literals as `print!`/`println!` args.
///
/// **Why is this bad?** Using literals as `println!` args is inefficient
/// (c.f., https://github.com/matthiaskrgr/rust-str-bench) and unnecessary
/// (i.e., just put the literal in the format string)
///
/// **Example:**
/// ```rust
/// println!("{}", "foo");
/// ```
declare_clippy_lint! {
    pub PRINT_LITERAL,
    style,
    "printing a literal with a format string"
}

#[derive(Copy, Clone, Debug)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(PRINT_WITH_NEWLINE, PRINTLN_EMPTY_STRING, PRINT_STDOUT, USE_DEBUG, PRINT_LITERAL)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if_chain! {
            if let ExprCall(ref fun, ref args) = expr.node;
            if let ExprPath(ref qpath) = fun.node;
            if let Some(fun_id) = opt_def_id(resolve_node(cx, qpath, fun.hir_id));
            then {

                // Search for `std::io::_print(..)` which is unique in a
                // `print!` expansion.
                if match_def_path(cx.tcx, fun_id, &paths::IO_PRINT) {
                    if let Some(span) = is_expn_of(expr.span, "print") {
                        // `println!` uses `print!`.
                        let (span, name) = match is_expn_of(span, "println") {
                            Some(span) => (span, "println"),
                            None => (span, "print"),
                        };

                        span_lint(cx, PRINT_STDOUT, span, &format!("use of `{}!`", name));

                        // Check for literals in the print!/println! args
                        // Also, ensure the format string is `{}` with no special options, like `{:X}`
                        check_print_args_for_literal(cx, args);

                        if_chain! {
                            // ensure we're calling Arguments::new_v1
                            if args.len() == 1;
                            if let ExprCall(ref args_fun, ref args_args) = args[0].node;
                            if let ExprPath(ref qpath) = args_fun.node;
                            if let Some(const_def_id) = opt_def_id(resolve_node(cx, qpath, args_fun.hir_id));
                            if match_def_path(cx.tcx, const_def_id, &paths::FMT_ARGUMENTS_NEWV1);
                            if args_args.len() == 2;
                            if let ExprAddrOf(_, ref match_expr) = args_args[1].node;
                            if let ExprMatch(ref args, _, _) = match_expr.node;
                            if let ExprTup(ref args) = args.node;
                            if let Some((fmtstr, fmtlen)) = get_argument_fmtstr_parts(&args_args[0]);
                            then {
                                match name {
                                    "print" => check_print(cx, span, args, fmtstr, fmtlen),
                                    "println" => check_println(cx, span, fmtstr, fmtlen),
                                    _ => (),
                                }
                            }
                        }
                    }
                }
                // Search for something like
                // `::std::fmt::ArgumentV1::new(__arg0, ::std::fmt::Debug::fmt)`
                else if args.len() == 2 && match_def_path(cx.tcx, fun_id, &paths::FMT_ARGUMENTV1_NEW) {
                    if let ExprPath(ref qpath) = args[1].node {
                        if let Some(def_id) = opt_def_id(cx.tables.qpath_def(qpath, args[1].hir_id)) {
                            if match_def_path(cx.tcx, def_id, &paths::DEBUG_FMT_METHOD)
                                    && !is_in_debug_impl(cx, expr) && is_expn_of(expr.span, "panic").is_none() {
                                span_lint(cx, USE_DEBUG, args[0].span, "use of `Debug`-based formatting");
                            }
                        }
                    }
                }
            }
        }
    }
}

// Check for literals in print!/println! args
// ensuring the format string for the literal is `DISPLAY_FMT_METHOD`
// e.g., `println!("... {} ...", "foo")`
//                                ^ literal in `println!`
fn check_print_args_for_literal<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    args: &HirVec<Expr>
) {
    if_chain! {
        if args.len() == 1;
        if let ExprCall(_, ref args_args) = args[0].node;
        if args_args.len() > 1;
        if let ExprAddrOf(_, ref match_expr) = args_args[1].node;
        if let ExprMatch(ref matchee, ref arms, _) = match_expr.node;
        if let ExprTup(ref tup) = matchee.node;
        if arms.len() == 1;
        if let ExprArray(ref arm_body_exprs) = arms[0].body.node;
        then {
            // it doesn't matter how many args there are in the `print!`/`println!`,
            // if there's one literal, we should warn the user
            for (idx, tup_arg) in tup.iter().enumerate() {
                if_chain! {
                    // first, make sure we're dealing with a literal (i.e., an ExprLit)
                    if let ExprAddrOf(_, ref tup_val) = tup_arg.node;
                    if let ExprLit(_) = tup_val.node;

                    // next, check the corresponding match arm body to ensure
                    // this is "{}", or DISPLAY_FMT_METHOD
                    if let ExprCall(_, ref body_args) = arm_body_exprs[idx].node;
                    if body_args.len() == 2;
                    if let ExprPath(ref body_qpath) = body_args[1].node;
                    if let Some(fun_def_id) = opt_def_id(resolve_node(cx, body_qpath, body_args[1].hir_id));
                    if match_def_path(cx.tcx, fun_def_id, &paths::DISPLAY_FMT_METHOD) ||
                       match_def_path(cx.tcx, fun_def_id, &paths::DEBUG_FMT_METHOD);
                    then {
                        span_lint(cx, PRINT_LITERAL, tup_val.span, "printing a literal with an empty format string");
                    }
                }
            }
        }
    }
}

// Check for print!("... \n", ...).
fn check_print<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    span: Span,
    args: &HirVec<Expr>,
    fmtstr: InternedString,
    fmtlen: usize,
) {
    if_chain! {
        // check the final format string part
        if let Some('\n') = fmtstr.chars().last();

        // "foo{}bar" is made into two strings + one argument,
        // if the format string starts with `{}` (eg. "{}foo"),
        // the string array is prepended an empty string "".
        // We only want to check the last string after any `{}`:
        if args.len() < fmtlen;
        then {
            span_lint(cx, PRINT_WITH_NEWLINE, span,
                      "using `print!()` with a format string that ends in a \
                       newline, consider using `println!()` instead");
        }
    }
}

/// Check for println!("")
fn check_println<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, span: Span, fmtstr: InternedString, fmtlen: usize) {
    if_chain! {
        // check that the string is empty
        if fmtlen == 1;
        if fmtstr.deref() == "\n";

        // check the presence of that string
        if let Ok(snippet) = cx.sess().codemap().span_to_snippet(span);
        if snippet.contains("\"\"");
        then {
            span_lint_and_sugg(
                cx,
                PRINT_WITH_NEWLINE,
                span,
                "using `println!(\"\")`",
                "replace it with",
                "println!()".to_string(),
            );
         }
    }
}

fn is_in_debug_impl(cx: &LateContext, expr: &Expr) -> bool {
    let map = &cx.tcx.hir;

    // `fmt` method
    if let Some(NodeImplItem(item)) = map.find(map.get_parent(expr.id)) {
        // `Debug` impl
        if let Some(NodeItem(item)) = map.find(map.get_parent(item.id)) {
            if let ItemImpl(_, _, _, _, Some(ref tr), _, _) = item.node {
                return match_path(&tr.path, &["Debug"]);
            }
        }
    }

    false
}

/// Returns the slice of format string parts in an `Arguments::new_v1` call.
fn get_argument_fmtstr_parts(expr: &Expr) -> Option<(InternedString, usize)> {
    if_chain! {
        if let ExprAddrOf(_, ref expr) = expr.node; // &["…", "…", …]
        if let ExprArray(ref exprs) = expr.node;
        if let Some(expr) = exprs.last();
        if let ExprLit(ref lit) = expr.node;
        if let LitKind::Str(ref lit, _) = lit.node;
        then {
            return Some((lit.as_str(), exprs.len()));
        }
    }
    None
}
