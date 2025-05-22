use clippy_utils::diagnostics::span_lint;
use clippy_utils::ty::{get_type_diagnostic_name, implements_trait};
use clippy_utils::{higher, sym};
use rustc_hir::{BorrowKind, Closure, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::Symbol;

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
    /// ```no_run
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
const HEURISTICS: [(Symbol, usize, Heuristic, Finiteness); 19] = [
    (sym::zip, 1, All, Infinite),
    (sym::chain, 1, Any, Infinite),
    (sym::cycle, 0, Always, Infinite),
    (sym::map, 1, First, Infinite),
    (sym::by_ref, 0, First, Infinite),
    (sym::cloned, 0, First, Infinite),
    (sym::rev, 0, First, Infinite),
    (sym::inspect, 0, First, Infinite),
    (sym::enumerate, 0, First, Infinite),
    (sym::peekable, 1, First, Infinite),
    (sym::fuse, 0, First, Infinite),
    (sym::skip, 1, First, Infinite),
    (sym::skip_while, 0, First, Infinite),
    (sym::filter, 1, First, Infinite),
    (sym::filter_map, 1, First, Infinite),
    (sym::flat_map, 1, First, Infinite),
    (sym::unzip, 0, First, Infinite),
    (sym::take_while, 1, First, MaybeInfinite),
    (sym::scan, 2, First, MaybeInfinite),
];

fn is_infinite(cx: &LateContext<'_>, expr: &Expr<'_>) -> Finiteness {
    match expr.kind {
        ExprKind::MethodCall(method, receiver, args, _) => {
            for &(name, len, heuristic, cap) in &HEURISTICS {
                if method.ident.name == name && args.len() == len {
                    return (match heuristic {
                        Always => Infinite,
                        First => is_infinite(cx, receiver),
                        Any => is_infinite(cx, receiver).or(is_infinite(cx, &args[0])),
                        All => is_infinite(cx, receiver).and(is_infinite(cx, &args[0])),
                    })
                    .and(cap);
                }
            }
            if method.ident.name == sym::flat_map
                && args.len() == 1
                && let ExprKind::Closure(&Closure { body, .. }) = args[0].kind
            {
                let body = cx.tcx.hir_body(body);
                return is_infinite(cx, body.value);
            }
            Finite
        },
        ExprKind::Block(block, _) => block.expr.as_ref().map_or(Finite, |e| is_infinite(cx, e)),
        ExprKind::AddrOf(BorrowKind::Ref, _, e) => is_infinite(cx, e),
        ExprKind::Call(path, _) => {
            if let ExprKind::Path(ref qpath) = path.kind {
                cx.qpath_res(qpath, path.hir_id)
                    .opt_def_id()
                    .is_some_and(|id| cx.tcx.is_diagnostic_item(sym::iter_repeat, id))
                    .into()
            } else {
                Finite
            }
        },
        ExprKind::Struct(..) => higher::Range::hir(expr).is_some_and(|r| r.end.is_none()).into(),
        _ => Finite,
    }
}

/// the names and argument lengths of methods that *may* exhaust their
/// iterators
const POSSIBLY_COMPLETING_METHODS: [(Symbol, usize); 6] = [
    (sym::find, 1),
    (sym::rfind, 1),
    (sym::position, 1),
    (sym::rposition, 1),
    (sym::any, 1),
    (sym::all, 1),
];

/// the names and argument lengths of methods that *always* exhaust
/// their iterators
const COMPLETING_METHODS: [(Symbol, usize); 12] = [
    (sym::count, 0),
    (sym::fold, 2),
    (sym::for_each, 1),
    (sym::partition, 1),
    (sym::max, 0),
    (sym::max_by, 1),
    (sym::max_by_key, 1),
    (sym::min, 0),
    (sym::min_by, 1),
    (sym::min_by_key, 1),
    (sym::sum, 0),
    (sym::product, 0),
];

fn complete_infinite_iter(cx: &LateContext<'_>, expr: &Expr<'_>) -> Finiteness {
    match expr.kind {
        ExprKind::MethodCall(method, receiver, args, _) => {
            let method_str = method.ident.name;
            for &(name, len) in &COMPLETING_METHODS {
                if method_str == name && args.len() == len {
                    return is_infinite(cx, receiver);
                }
            }
            for &(name, len) in &POSSIBLY_COMPLETING_METHODS {
                if method_str == name && args.len() == len {
                    return MaybeInfinite.and(is_infinite(cx, receiver));
                }
            }
            if method.ident.name == sym::last && args.is_empty() {
                let not_double_ended = cx
                    .tcx
                    .get_diagnostic_item(sym::DoubleEndedIterator)
                    .is_some_and(|id| !implements_trait(cx, cx.typeck_results().expr_ty(receiver), id, &[]));
                if not_double_ended {
                    return is_infinite(cx, receiver);
                }
            } else if method.ident.name == sym::collect {
                let ty = cx.typeck_results().expr_ty(expr);
                if matches!(
                    get_type_diagnostic_name(cx, ty),
                    Some(
                        sym::BinaryHeap
                            | sym::BTreeMap
                            | sym::BTreeSet
                            | sym::HashMap
                            | sym::HashSet
                            | sym::LinkedList
                            | sym::Vec
                            | sym::VecDeque,
                    )
                ) {
                    return is_infinite(cx, receiver);
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
