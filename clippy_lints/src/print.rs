use rustc::hir::*;
use rustc::hir::map::Node::{NodeItem, NodeImplItem};
use rustc::lint::*;
use utils::paths;
use utils::{is_expn_of, match_def_path, resolve_node, span_lint, match_path_old};
use format::get_argument_fmtstr_parts;

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
        lint_array!(PRINT_WITH_NEWLINE, PRINT_STDOUT, USE_DEBUG)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if_let_chain! {[
            let ExprCall(ref fun, ref args) = expr.node,
            let ExprPath(ref qpath) = fun.node,
        ], {
            let fun = resolve_node(cx, qpath, fun.id);
            let fun_id = fun.def_id();

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

                    // Check print! with format string ending in "\n".
                    if_let_chain!{[
                        name == "print",

                        // ensure we're calling Arguments::new_v1
                        args.len() == 1,
                        let ExprCall(ref args_fun, ref args_args) = args[0].node,
                        let ExprPath(ref qpath) = args_fun.node,
                        match_def_path(cx.tcx,
                                       resolve_node(cx, qpath, args_fun.id).def_id(),
                                       &paths::FMT_ARGUMENTS_NEWV1),
                        args_args.len() == 2,
                        let ExprAddrOf(_, ref match_expr) = args_args[1].node,
                        let ExprMatch(ref args, _, _) = match_expr.node,
                        let ExprTup(ref args) = args.node,

                        // collect the format string parts and check the last one
                        let Some(fmtstrs) = get_argument_fmtstr_parts(cx, &args_args[0]),
                        let Some(last_str) = fmtstrs.last(),
                        let Some('\n') = last_str.chars().last(),

                        // "foo{}bar" is made into two strings + one argument,
                        // if the format string starts with `{}` (eg. "{}foo"),
                        // the string array is prepended an empty string "".
                        // We only want to check the last string after any `{}`:
                        args.len() < fmtstrs.len(),
                    ], {
                        span_lint(cx, PRINT_WITH_NEWLINE, span,
                                  "using `print!()` with a format string that ends in a \
                                   newline, consider using `println!()` instead");
                    }}
                }
            }
            // Search for something like
            // `::std::fmt::ArgumentV1::new(__arg0, ::std::fmt::Debug::fmt)`
            else if args.len() == 2 && match_def_path(cx.tcx, fun_id, &paths::FMT_ARGUMENTV1_NEW) {
                if let ExprPath(ref qpath) = args[1].node {
                    let def_id = cx.tables.qpath_def(qpath, args[1].id).def_id();
                    if match_def_path(cx.tcx, def_id, &paths::DEBUG_FMT_METHOD) && !is_in_debug_impl(cx, expr) &&
                       is_expn_of(expr.span, "panic").is_none() {
                        span_lint(cx, USE_DEBUG, args[0].span, "use of `Debug`-based formatting");
                    }
                }
            }
        }}
    }
}

fn is_in_debug_impl(cx: &LateContext, expr: &Expr) -> bool {
    let map = &cx.tcx.hir;

    // `fmt` method
    if let Some(NodeImplItem(item)) = map.find(map.get_parent(expr.id)) {
        // `Debug` impl
        if let Some(NodeItem(item)) = map.find(map.get_parent(item.id)) {
            if let ItemImpl(_, _, _, _, Some(ref tr), _, _) = item.node {
                return match_path_old(&tr.path, &["Debug"]);
            }
        }
    }

    false
}
