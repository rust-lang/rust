use rustc::lint::{Context, LintPass, LintArray};
use rustc_front::hir::*;
use syntax::codemap::Spanned;
use syntax::ptr::P;
use std::cmp::PartialOrd;
use std::cmp::Ordering::*;

use consts::{Constant, constant};
use utils::{match_path, span_lint};
use self::MinMax::{Min, Max};

declare_lint!(pub MIN_MAX, Deny,
    "`min(_, max(_, _))` (or vice versa) with bounds clamping the result \
    to a constant");

#[allow(missing_copy_implementations)]
pub struct MinMaxPass;

impl LintPass for MinMaxPass {
    fn get_lints(&self) -> LintArray {
       lint_array!(MIN_MAX)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        if let Some((outer_max, outer_c, oe)) = min_max(cx, expr) {
            if let Some((inner_max, inner_c, _)) = min_max(cx, oe) {
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

fn min_max<'e>(cx: &Context, expr: &'e Expr) ->
        Option<(MinMax, Constant, &'e Expr)> {
    match expr.node {
        ExprMethodCall(Spanned{node: ref ident, ..}, _, ref args) => {
            let name = ident.name;
            if name == "min" {
                fetch_const(cx, args, Min)
            } else {
                if name == "max" {
                    fetch_const(cx, args, Max)
                } else {
                    None
                }
            }
        },
        ExprCall(ref path, ref args) => {
            if let &ExprPath(None, ref path) = &path.node {
                if match_path(path, &["min"]) {
                    fetch_const(cx, args, Min)
                } else {
                    if match_path(path, &["max"]) {
                        fetch_const(cx, args, Max)
                    } else {
                        None
                    }
                }
            } else { None }
         },
         _ => None,
     }
 }

fn fetch_const<'e>(cx: &Context, args: &'e Vec<P<Expr>>, m: MinMax) ->
        Option<(MinMax, Constant, &'e Expr)> {
    if args.len() != 2 { return None }
    if let Some((c, _)) = constant(cx, &args[0]) {
        if let None = constant(cx, &args[1]) { // otherwise ignore
            Some((m, c, &args[1]))
        } else { None }
    } else {
        if let Some((c, _)) = constant(cx, &args[1]) {
            Some((m, c, &args[0]))
        } else { None }
    }
}
