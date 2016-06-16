use rustc::hir::*;
use rustc::hir::map::Node::{NodeItem, NodeImplItem};
use rustc::lint::*;
use utils::paths;
use utils::{is_expn_of, match_path, span_lint};

/// **What it does:** This lint warns whenever you print on *stdout*. The purpose of this lint is to catch debugging remnants.
///
/// **Why is this bad?** People often print on *stdout* while debugging an application and might
/// forget to remove those prints afterward.
///
/// **Known problems:** Only catches `print!` and `println!` calls.
///
/// **Example:** `println!("Hello world!");`
declare_lint! {
    pub PRINT_STDOUT,
    Allow,
    "printing on stdout"
}

/// **What it does:** This lint warns whenever you use `Debug` formatting. The purpose of this lint is to catch debugging remnants.
///
/// **Why is this bad?** The purpose of the `Debug` trait is to facilitate debugging Rust code. It
/// should not be used in in user-facing output.
///
/// **Example:** `println!("{:?}", foo);`
declare_lint! {
    pub USE_DEBUG,
    Allow,
    "use `Debug`-based formatting"
}

#[derive(Copy, Clone, Debug)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(PRINT_STDOUT, USE_DEBUG)
    }
}

impl LateLintPass for Pass {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if let ExprCall(ref fun, ref args) = expr.node {
            if let ExprPath(_, ref path) = fun.node {
                // Search for `std::io::_print(..)` which is unique in a
                // `print!` expansion.
                if match_path(path, &paths::IO_PRINT) {
                    if let Some(span) = is_expn_of(cx, expr.span, "print") {
                        // `println!` uses `print!`.
                        let (span, name) = match is_expn_of(cx, span, "println") {
                            Some(span) => (span, "println"),
                            None => (span, "print"),
                        };

                        span_lint(cx, PRINT_STDOUT, span, &format!("use of `{}!`", name));
                    }
                }
                // Search for something like
                // `::std::fmt::ArgumentV1::new(__arg0, ::std::fmt::Debug::fmt)`
                else if args.len() == 2 && match_path(path, &paths::FMT_ARGUMENTV1_NEW) {
                    if let ExprPath(None, ref path) = args[1].node {
                        if match_path(path, &paths::DEBUG_FMT_METHOD) && !is_in_debug_impl(cx, expr) &&
                           is_expn_of(cx, expr.span, "panic").is_none() {
                            span_lint(cx, USE_DEBUG, args[0].span, "use of `Debug`-based formatting");
                        }
                    }
                }
            }
        }
    }
}

fn is_in_debug_impl(cx: &LateContext, expr: &Expr) -> bool {
    let map = &cx.tcx.map;

    // `fmt` method
    if let Some(NodeImplItem(item)) = map.find(map.get_parent(expr.id)) {
        // `Debug` impl
        if let Some(NodeItem(item)) = map.find(map.get_parent(item.id)) {
            if let ItemImpl(_, _, _, Some(ref tr), _, _) = item.node {
                return match_path(&tr.path, &["Debug"]);
            }
        }
    }

    false
}
