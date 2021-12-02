use clippy_utils::diagnostics::span_lint;
use clippy_utils::ty::{implements_trait, is_type_diagnostic_item};
use clippy_utils::{get_trait_def_id, higher, is_qpath_def_path, paths};
use rustc_hir::{BorrowKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::{sym, Symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for iteration that is guaranteed to be infinite.
    ///
    /// ### Why is this bad?
    /// While there may be places where this is acceptable
    /// (e.g., in event streams), in most cases this is simply an error.
    ///
    /// ### Example
    /// ```no_run
    /// use std::iter;
    ///
    /// iter::repeat(1_u8).collect::<Vec<_>>();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub INFINITE_ITER,
    correctness,
    "infinite iteration"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for iteration that may be infinite.
    ///
    /// ### Why is this bad?
    /// While there may be places where this is acceptable
    /// (e.g., in event streams), in most cases this is simply an error.
    ///
    /// ### Known problems
    /// The code may have a condition to stop iteration, but
    /// this lint is not clever enough to analyze it.
    ///
    /// ### Example
    /// ```rust
    /// let infinite_iter = 0..;
    /// [0..].iter().zip(infinite_iter.take_while(|x| *x > 5));
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub MAYBE_INFINITE_ITER,
    pedantic,
    "possible infinite iteration"
}

declare_lint_pass!(InfiniteIter => [INFINITE_ITER, MAYBE_INFINITE_ITER]);

impl<'tcx> LateLintPass<'tcx> for InfiniteIter {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let (lint, msg) = match complete_infinite_iter(cx, expr) {
            Infinite => (INFINITE_ITER, "infinite iteration detected"),
            MaybeInfinite => (MAYBE_INFINITE_ITER, "possible infinite iteration detected"),
            Finite => {
                return;
            },
        };
        span_lint(cx, lint, expr.span, msg);
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Finiteness {
    Infinite,
    MaybeInfinite,
    Finite,
}

use self::Finiteness::{Finite, Infinite, MaybeInfinite};

impl Finiteness {
    #[must_use]
    fn and(self, b: Self) -> Self {
        match (self, b) {
            (Finite, _) | (_, Finite) => Finite,
            (MaybeInfinite, _) | (_, MaybeInfinite) => MaybeInfinite,
            _ => Infinite,
        }
    }

    #[must_use]
    fn or(self, b: Self) -> Self {
        match (self, b) {
            (Infinite, _) | (_, Infinite) => Infinite,
            (MaybeInfinite, _) | (_, MaybeInfinite) => MaybeInfinite,
            _ => Finite,
        }
    }
}

impl From<bool> for Finiteness {
    #[must_use]
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
    All,
}

use self::Heuristic::{All, Always, Any, First};

/// a slice of (method name, number of args, heuristic, bounds) tuples
/// that will be used to determine whether the method in question
/// returns an infinite or possibly infinite iterator. The finiteness
/// is an upper bound, e.g., some methods can return a possibly
/// infinite iterator at worst, e.g., `take_while`.
const HEURISTICS: [(&str, usize, Heuristic, Finiteness); 19] = [
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
    ("scan", 3, First, MaybeInfinite),
];

fn is_infinite(cx: &LateContext<'_>, expr: &Expr<'_>) -> Finiteness {
    match expr.kind {
        ExprKind::MethodCall(method, _, args, _) => {
            for &(name, len, heuristic, cap) in &HEURISTICS {
                if method.ident.name.as_str() == name && args.len() == len {
                    return (match heuristic {
                        Always => Infinite,
                        First => is_infinite(cx, &args[0]),
                        Any => is_infinite(cx, &args[0]).or(is_infinite(cx, &args[1])),
                        All => is_infinite(cx, &args[0]).and(is_infinite(cx, &args[1])),
                    })
                    .and(cap);
                }
            }
            if method.ident.name == sym!(flat_map) && args.len() == 2 {
                if let ExprKind::Closure(_, _, body_id, _, _) = args[1].kind {
                    let body = cx.tcx.hir().body(body_id);
                    return is_infinite(cx, &body.value);
                }
            }
            Finite
        },
        ExprKind::Block(block, _) => block.expr.as_ref().map_or(Finite, |e| is_infinite(cx, e)),
        ExprKind::Box(e) | ExprKind::AddrOf(BorrowKind::Ref, _, e) => is_infinite(cx, e),
        ExprKind::Call(path, _) => {
            if let ExprKind::Path(ref qpath) = path.kind {
                is_qpath_def_path(cx, qpath, path.hir_id, &paths::ITER_REPEAT).into()
            } else {
                Finite
            }
        },
        ExprKind::Struct(..) => higher::Range::hir(expr).map_or(false, |r| r.end.is_none()).into(),
        _ => Finite,
    }
}

/// the names and argument lengths of methods that *may* exhaust their
/// iterators
const POSSIBLY_COMPLETING_METHODS: [(&str, usize); 6] = [
    ("find", 2),
    ("rfind", 2),
    ("position", 2),
    ("rposition", 2),
    ("any", 2),
    ("all", 2),
];

/// the names and argument lengths of methods that *always* exhaust
/// their iterators
const COMPLETING_METHODS: [(&str, usize); 12] = [
    ("count", 1),
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
    ("product", 1),
];

/// the paths of types that are known to be infinitely allocating
const INFINITE_COLLECTORS: &[Symbol] = &[
    sym::BinaryHeap,
    sym::BTreeMap,
    sym::BTreeSet,
    sym::HashMap,
    sym::HashSet,
    sym::LinkedList,
    sym::Vec,
    sym::VecDeque,
];

fn complete_infinite_iter(cx: &LateContext<'_>, expr: &Expr<'_>) -> Finiteness {
    match expr.kind {
        ExprKind::MethodCall(method, _, args, _) => {
            for &(name, len) in &COMPLETING_METHODS {
                if method.ident.name.as_str() == name && args.len() == len {
                    return is_infinite(cx, &args[0]);
                }
            }
            for &(name, len) in &POSSIBLY_COMPLETING_METHODS {
                if method.ident.name.as_str() == name && args.len() == len {
                    return MaybeInfinite.and(is_infinite(cx, &args[0]));
                }
            }
            if method.ident.name == sym!(last) && args.len() == 1 {
                let not_double_ended = get_trait_def_id(cx, &paths::DOUBLE_ENDED_ITERATOR).map_or(false, |id| {
                    !implements_trait(cx, cx.typeck_results().expr_ty(&args[0]), id, &[])
                });
                if not_double_ended {
                    return is_infinite(cx, &args[0]);
                }
            } else if method.ident.name == sym!(collect) {
                let ty = cx.typeck_results().expr_ty(expr);
                if INFINITE_COLLECTORS
                    .iter()
                    .any(|diag_item| is_type_diagnostic_item(cx, ty, *diag_item))
                {
                    return is_infinite(cx, &args[0]);
                }
            }
        },
        ExprKind::Binary(op, l, r) => {
            if op.node.is_comparison() {
                return is_infinite(cx, l).and(is_infinite(cx, r)).and(MaybeInfinite);
            }
        }, // TODO: ExprKind::Loop + Match
        _ => (),
    }
    Finite
}
