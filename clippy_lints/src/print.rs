use std::ops::Deref;
use rustc::hir::*;
use rustc::hir::map::Node::{NodeImplItem, NodeItem};
use rustc::lint::*;
use syntax::ast::LitKind;
use syntax::symbol::InternedString;
use syntax_pos::Span;
use utils::{is_expn_of, match_def_path, match_path, resolve_node, span_lint};
use utils::{opt_def_id, paths};

/// **What it does:** This lint warns when you using `println!("")` to
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
declare_lint! {
    pub PRINTLN_EMPTY_STRING,
    Warn,
    "using `print!()` with a format string that ends in a newline"
}

/// **What it does:** This lint warns when you using `print!()` with a format
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
declare_lint! {
    pub PRINT_WITH_NEWLINE,
    Warn,
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
declare_lint! {
    pub PRINT_STDOUT,
    Allow,
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
declare_lint! {
    pub USE_DEBUG,
    Allow,
    "use of `Debug`-based formatting"
}

#[derive(Copy, Clone, Debug)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(PRINT_WITH_NEWLINE, PRINTLN_EMPTY_STRING, PRINT_STDOUT, USE_DEBUG)
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
            span_lint(cx, PRINT_WITH_NEWLINE, span,
                      "using `println!(\"\")`, consider using `println!()` instead");
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
