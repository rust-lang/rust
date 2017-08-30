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
            Infinite => (INFINITE_ITER, "infinite iteration detected"),
            MaybeInfinite => (MAYBE_INFINITE_ITER,
                        "possible infinite iteration detected"),
            Finite => { return; }
        };
        span_lint(cx, lint, expr.span, msg)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Finiteness {
    Infinite,
    MaybeInfinite,
    Finite
}

use self::Finiteness::{Infinite, MaybeInfinite, Finite};

impl Finiteness {
    fn and(self, b: Self) -> Self {
        match (self, b) {
            (Finite, _) | (_, Finite) => Finite,
            (MaybeInfinite, _) | (_, MaybeInfinite) => MaybeInfinite,
            _ => Infinite
        }
    }

    fn or(self, b: Self) -> Self {
        match (self, b) {
            (Infinite, _) | (_, Infinite) => Infinite,
            (MaybeInfinite, _) | (_, MaybeInfinite) => MaybeInfinite,
            _ => Finite
        }
    }
}

impl From<bool> for Finiteness {
    fn from(b: bool) -> Self {
        if b { Infinite } else { Finite }
    }
}

/// This tells us what to look for to know if the iterator returned by
/// this method is infinite
#[derive(Copy, Clone)]
enum Heuristic {
    /// infinite no matter what
    Always,
    /// infinite if the first argument is
    First,
    /// infinite if any of the supplied arguments is
    Any,
    /// infinite if all of the supplied arguments are
    All
}

use self::Heuristic::{Always, First, Any, All};

/// a slice of (method name, number of args, heuristic, bounds) tuples
/// that will be used to determine whether the method in question
/// returns an infinite or possibly infinite iterator. The finiteness
/// is an upper bound, e.g. some methods can return a possibly
/// infinite iterator at worst, e.g. `take_while`.
static HEURISTICS : &[(&str, usize, Heuristic, Finiteness)] = &[
    ("zip", 2, All, Infinite),
    ("chain", 2, Any, Infinite),
    ("cycle", 1, Always, Infinite),
    ("map", 2, First, Infinite),
    ("by_ref", 1, First, Infinite),
    ("cloned", 1, First, Infinite),
    ("rev", 1, First, Infinite),
    ("inspect", 1, First, Infinite),
    ("enumerate", 1, First, Infinite),
    ("peekable", 2, First, Infinite),
    ("fuse", 1, First, Infinite),
    ("skip", 2, First, Infinite),
    ("skip_while", 1, First, Infinite),
    ("filter", 2, First, Infinite),
    ("filter_map", 2, First, Infinite),
    ("flat_map", 2, First, Infinite),
    ("unzip", 1, First, Infinite),
    ("take_while", 2, First, MaybeInfinite),
    ("scan", 3, First, MaybeInfinite)
];

fn is_infinite(cx: &LateContext, expr: &Expr) -> Finiteness {
    match expr.node {
        ExprMethodCall(ref method, _, ref args) => {
            for &(name, len, heuristic, cap) in HEURISTICS.iter() {
                if method.name == name && args.len() == len {
                    return (match heuristic {
                        Always => Infinite,
                        First => is_infinite(cx, &args[0]),
                        Any => is_infinite(cx, &args[0]).or(is_infinite(cx, &args[1])),
                        All => is_infinite(cx, &args[0]).and(is_infinite(cx, &args[1])),
                    }).and(cap);
                }
            }
            if method.name == "flat_map" && args.len() == 2 {
                if let ExprClosure(_, _, body_id, _, _) = args[1].node {
                    let body = cx.tcx.hir.body(body_id);
                    return is_infinite(cx, &body.value);
                }
            }
            Finite
        },
        ExprBlock(ref block) =>
            block.expr.as_ref().map_or(Finite, |e| is_infinite(cx, e)),
        ExprBox(ref e) | ExprAddrOf(_, ref e) => is_infinite(cx, e),
        ExprCall(ref path, _) => {
            if let ExprPath(ref qpath) = path.node {
                match_qpath(qpath, &paths::REPEAT).into()
            } else { Finite }
        },
        ExprStruct(..) => {
            higher::range(expr).map_or(false, |r| r.end.is_none()).into()
        },
        _ => Finite
    }
}

/// the names and argument lengths of methods that *may* exhaust their
/// iterators
static POSSIBLY_COMPLETING_METHODS : &[(&str, usize)] = &[
    ("find", 2),
    ("rfind", 2),
    ("position", 2),
    ("rposition", 2),
    ("any", 2),
    ("all", 2)
];

/// the names and argument lengths of methods that *always* exhaust
/// their iterators
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

fn complete_infinite_iter(cx: &LateContext, expr: &Expr) -> Finiteness {
    match expr.node {
        ExprMethodCall(ref method, _, ref args) => {
            for &(name, len) in COMPLETING_METHODS.iter() {
                if method.name == name && args.len() == len {
                    return is_infinite(cx, &args[0]);
                }
            }
            for &(name, len) in POSSIBLY_COMPLETING_METHODS.iter() {
                if method.name == name && args.len() == len {
                    return MaybeInfinite.and(is_infinite(cx, &args[0]));
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
                return is_infinite(cx, l).and(is_infinite(cx, r)).and(MaybeInfinite)
            }
        }, //TODO: ExprLoop + Match
        _ => ()
    }
    Finite
}
