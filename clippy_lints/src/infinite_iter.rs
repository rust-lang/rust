use rustc::hir::*;
use rustc::lint::*;
use utils::{get_trait_def_id, implements_trait, higher, match_qpath, paths, span_lint};

/// **What it does:** Checks for iteration that is guaranteed to be infinite.
///
/// **Why is this bad?** While there may be places where this is acceptable
/// (e.g. in event streams), in most cases this is simply an error.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// repeat(1_u8).iter().collect::<Vec<_>>()
/// ```
declare_lint! {
    pub INFINITE_ITER,
    Warn,
    "infinite iteration"
}

/// **What it does:** Checks for iteration that may be infinite.
///
/// **Why is this bad?** While there may be places where this is acceptable
/// (e.g. in event streams), in most cases this is simply an error.
///
/// **Known problems:** The code may have a condition to stop iteration, but
/// this lint is not clever enough to analyze it.
///
/// **Example:**
/// ```rust
/// [0..].iter().zip(infinite_iter.take_while(|x| x > 5))
/// ```
declare_lint! {
    pub MAYBE_INFINITE_ITER,
    Allow,
    "possible infinite iteration"
}

#[derive(Copy, Clone)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(INFINITE_ITER, MAYBE_INFINITE_ITER)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        let (lint, msg) = match complete_infinite_iter(cx, expr) {
            True => (INFINITE_ITER, "infinite iteration detected"),
            Unknown => (MAYBE_INFINITE_ITER,
                        "possible infinite iteration detected"),
            False => { return; }
        };
        span_lint(cx, lint, expr.span, msg)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum TriState {
    True,
    Unknown,
    False
}

use self::TriState::{True, Unknown, False};

impl TriState {
    fn and(self, b: Self) -> Self {
        match (self, b) {
            (False, _) | (_, False) => False,
            (Unknown, _) | (_, Unknown) => Unknown,
            _ => True
        }
    }

    fn or(self, b: Self) -> Self {
        match (self, b) {
            (True, _) | (_, True) => True,
            (Unknown, _) | (_, Unknown) => Unknown,
            _ => False
        }
    }
}

impl From<bool> for TriState {
    fn from(b: bool) -> Self {
        if b { True } else { False }
    }
}

#[derive(Copy, Clone)]
enum Heuristic {
    Always,
    First,
    Any,
    All
}

use self::Heuristic::{Always, First, Any, All};

// here we use the `TriState` as (Finite, Possible Infinite, Infinite)
static HEURISTICS : &[(&str, usize, Heuristic, TriState)] = &[
    ("zip", 2, All, True),
    ("chain", 2, Any, True),
    ("cycle", 1, Always, True),
    ("map", 2, First, True),
    ("by_ref", 1, First, True),
    ("cloned", 1, First, True),
    ("rev", 1, First, True),
    ("inspect", 1, First, True),
    ("enumerate", 1, First, True),
    ("peekable", 2, First, True),
    ("fuse", 1, First, True),
    ("skip", 2, First, True),
    ("skip_while", 1, First, True),
    ("filter", 2, First, True),
    ("filter_map", 2, First, True),
    ("flat_map", 2, First, True),
    ("unzip", 1, First, True),
    ("take_while", 2, First, Unknown),
    ("scan", 3, First, Unknown)
];

fn is_infinite(cx: &LateContext, expr: &Expr) -> TriState {
    match expr.node {
        ExprMethodCall(ref method, _, ref args) => {
            for &(name, len, heuristic, cap) in HEURISTICS.iter() {
                if method.name == name && args.len() == len {
                    return (match heuristic {
                        Always => True,
                        First => is_infinite(cx, &args[0]),
                        Any => is_infinite(cx, &args[0]).or(is_infinite(cx, &args[1])),
                        All => is_infinite(cx, &args[0]).and(is_infinite(cx, &args[1])),
                    }).and(cap);
                }
            }
            if method.name == "flat_map" && args.len() == 2 {
                if let ExprClosure(_, _, body_id, _) = args[1].node {
                    let body = cx.tcx.hir.body(body_id);
                    return is_infinite(cx, &body.value);
                }
            }
            False
        },
        ExprBlock(ref block) =>
            block.expr.as_ref().map_or(False, |e| is_infinite(cx, e)),
        ExprBox(ref e) | ExprAddrOf(_, ref e) => is_infinite(cx, e),
        ExprCall(ref path, _) => {
            if let ExprPath(ref qpath) = path.node {
                match_qpath(qpath, &paths::REPEAT).into()
            } else { False }
        },
        ExprStruct(..) => {
            higher::range(expr).map_or(false, |r| r.end.is_none()).into()
        },
        _ => False
    }
}

static POSSIBLY_COMPLETING_METHODS : &[(&str, usize)] = &[
    ("find", 2),
    ("rfind", 2),
    ("position", 2),
    ("rposition", 2),
    ("any", 2),
    ("all", 2)
];

static COMPLETING_METHODS : &[(&str, usize)] = &[
    ("count", 1),
    ("collect", 1),
    ("fold", 3),
    ("for_each", 2),
    ("partition", 2),
    ("max", 1),
    ("max_by", 2),
    ("max_by_key", 2),
    ("min", 1),
    ("min_by", 2),
    ("min_by_key", 2),
    ("sum", 1),
    ("product", 1)
];

fn complete_infinite_iter(cx: &LateContext, expr: &Expr) -> TriState {
    match expr.node {
        ExprMethodCall(ref method, _, ref args) => {
            for &(name, len) in COMPLETING_METHODS.iter() {
                if method.name == name && args.len() == len {
                    return is_infinite(cx, &args[0]);
                }
            }
            for &(name, len) in POSSIBLY_COMPLETING_METHODS.iter() {
                if method.name == name && args.len() == len {
                    return Unknown.and(is_infinite(cx, &args[0]));
                }
            }
            if method.name == "last" && args.len() == 1 &&
                    get_trait_def_id(cx, &paths::DOUBLE_ENDED_ITERATOR).map_or(false,
                        |id| !implements_trait(cx,
                                               cx.tables.expr_ty(&args[0]),
                                               id,
                                               &[])) {
                return is_infinite(cx, &args[0]);
            }
        },
        ExprBinary(op, ref l, ref r) => {
            if op.node.is_comparison() {
                return is_infinite(cx, l).and(is_infinite(cx, r)).and(Unknown)
            }
        }, //TODO: ExprLoop + Match
        _ => ()
    }
    False
}
