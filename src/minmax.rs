use rustc::lint::{Context, LintPass, LintArray};
use rustc_front::hir::*;
use syntax::ptr::P;
use std::cmp::PartialOrd;
use std::cmp::Ordering::*;

use consts::{Constant, constant_simple};
use utils::{match_path, span_lint};
use self::MinMax::{Min, Max};

declare_lint!(pub MIN_MAX, Warn,
    "`min(_, max(_, _))` (or vice versa) with bounds clamping the result \
    to a constant");

#[allow(missing_copy_implementations)]
pub struct MinMaxPass;

impl LintPass for MinMaxPass {
    fn get_lints(&self) -> LintArray {
       lint_array!(MIN_MAX)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        if let Some((outer_max, outer_c, oe)) = min_max(expr) {
            if let Some((inner_max, inner_c, _)) = min_max(oe) {
                if outer_max == inner_max { return; }
                match (outer_max, outer_c.partial_cmp(&inner_c)) {
                    (_, None) | (Max, Some(Less)) | (Min, Some(Greater)) => (),
                    _ => {
                        span_lint(cx, MIN_MAX, expr.span,
                            "this min/max combination leads to constant result")
                    },
                }
            }
        }
    }
}

#[derive(PartialEq, Eq, Debug)]
enum MinMax {
    Min,
    Max,
}

fn min_max(expr: &Expr) -> Option<(MinMax, Constant, &Expr)> {
    if let ExprCall(ref path, ref args) = expr.node {
        if let ExprPath(None, ref path) = path.node {
            if match_path(path, &["std", "cmp", "min"]) {
                fetch_const(args, Min)
            } else {
                if match_path(path, &["std", "cmp", "max"]) {
                    fetch_const(args, Max)
                } else {
                    None
                }
            }
        } else { None }
    } else { None }
 }

fn fetch_const(args: &[P<Expr>], m: MinMax) ->
        Option<(MinMax, Constant, &Expr)> {
    if args.len() != 2 { return None }
    if let Some(c) = constant_simple(&args[0]) {
        if let None = constant_simple(&args[1]) { // otherwise ignore
            Some((m, c, &args[1]))
        } else { None }
    } else {
        if let Some(c) = constant_simple(&args[1]) {
            Some((m, c, &args[0]))
        } else { None }
    }
}
