use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};

use crate::utils::sym;
use crate::utils::{get_trait_def_id, higher, implements_trait, match_qpath, match_type, paths, span_lint};
use lazy_static::lazy_static;
use syntax::symbol::Symbol;

declare_clippy_lint! {
    /// **What it does:** Checks for iteration that is guaranteed to be infinite.
    ///
    /// **Why is this bad?** While there may be places where this is acceptable
    /// (e.g., in event streams), in most cases this is simply an error.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```no_run
    /// use std::iter;
    ///
    /// iter::repeat(1_u8).collect::<Vec<_>>();
    /// ```
    pub INFINITE_ITER,
    correctness,
    "infinite iteration"
}

declare_clippy_lint! {
    /// **What it does:** Checks for iteration that may be infinite.
    ///
    /// **Why is this bad?** While there may be places where this is acceptable
    /// (e.g., in event streams), in most cases this is simply an error.
    ///
    /// **Known problems:** The code may have a condition to stop iteration, but
    /// this lint is not clever enough to analyze it.
    ///
    /// **Example:**
    /// ```rust
    /// [0..].iter().zip(infinite_iter.take_while(|x| x > 5))
    /// ```
    pub MAYBE_INFINITE_ITER,
    pedantic,
    "possible infinite iteration"
}

declare_lint_pass!(InfiniteIter => [INFINITE_ITER, MAYBE_INFINITE_ITER]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for InfiniteIter {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        let (lint, msg) = match complete_infinite_iter(cx, expr) {
            Infinite => (INFINITE_ITER, "infinite iteration detected"),
            MaybeInfinite => (MAYBE_INFINITE_ITER, "possible infinite iteration detected"),
            Finite => {
                return;
            },
        };
        span_lint(cx, lint, expr.span, msg)
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
    fn and(self, b: Self) -> Self {
        match (self, b) {
            (Finite, _) | (_, Finite) => Finite,
            (MaybeInfinite, _) | (_, MaybeInfinite) => MaybeInfinite,
            _ => Infinite,
        }
    }

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
        if b {
            Infinite
        } else {
            Finite
        }
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

lazy_static! {
/// a slice of (method name, number of args, heuristic, bounds) tuples
/// that will be used to determine whether the method in question
/// returns an infinite or possibly infinite iterator. The finiteness
/// is an upper bound, e.g., some methods can return a possibly
/// infinite iterator at worst, e.g., `take_while`.
static ref HEURISTICS: [(Symbol, usize, Heuristic, Finiteness); 19] = [
    (*sym::zip, 2, All, Infinite),
    (*sym::chain, 2, Any, Infinite),
    (*sym::cycle, 1, Always, Infinite),
    (*sym::map, 2, First, Infinite),
    (*sym::by_ref, 1, First, Infinite),
    (*sym::cloned, 1, First, Infinite),
    (*sym::rev, 1, First, Infinite),
    (*sym::inspect, 1, First, Infinite),
    (*sym::enumerate, 1, First, Infinite),
    (*sym::peekable, 2, First, Infinite),
    (*sym::fuse, 1, First, Infinite),
    (*sym::skip, 2, First, Infinite),
    (*sym::skip_while, 1, First, Infinite),
    (*sym::filter, 2, First, Infinite),
    (*sym::filter_map, 2, First, Infinite),
    (*sym::flat_map, 2, First, Infinite),
    (*sym::unzip, 1, First, Infinite),
    (*sym::take_while, 2, First, MaybeInfinite),
    (*sym::scan, 3, First, MaybeInfinite),
];
}

fn is_infinite(cx: &LateContext<'_, '_>, expr: &Expr) -> Finiteness {
    match expr.node {
        ExprKind::MethodCall(ref method, _, ref args) => {
            for &(name, len, heuristic, cap) in HEURISTICS.iter() {
                if method.ident.name == name && args.len() == len {
                    return (match heuristic {
                        Always => Infinite,
                        First => is_infinite(cx, &args[0]),
                        Any => is_infinite(cx, &args[0]).or(is_infinite(cx, &args[1])),
                        All => is_infinite(cx, &args[0]).and(is_infinite(cx, &args[1])),
                    })
                    .and(cap);
                }
            }
            if method.ident.name == *sym::flat_map && args.len() == 2 {
                if let ExprKind::Closure(_, _, body_id, _, _) = args[1].node {
                    let body = cx.tcx.hir().body(body_id);
                    return is_infinite(cx, &body.value);
                }
            }
            Finite
        },
        ExprKind::Block(ref block, _) => block.expr.as_ref().map_or(Finite, |e| is_infinite(cx, e)),
        ExprKind::Box(ref e) | ExprKind::AddrOf(_, ref e) => is_infinite(cx, e),
        ExprKind::Call(ref path, _) => {
            if let ExprKind::Path(ref qpath) = path.node {
                match_qpath(qpath, &*paths::REPEAT).into()
            } else {
                Finite
            }
        },
        ExprKind::Struct(..) => higher::range(cx, expr).map_or(false, |r| r.end.is_none()).into(),
        _ => Finite,
    }
}

lazy_static! {
/// the names and argument lengths of methods that *may* exhaust their
/// iterators
static ref POSSIBLY_COMPLETING_METHODS: [(Symbol, usize); 6] = [
    (*sym::find, 2),
    (*sym::rfind, 2),
    (*sym::position, 2),
    (*sym::rposition, 2),
    (*sym::any, 2),
    (*sym::all, 2),
];

/// the names and argument lengths of methods that *always* exhaust
/// their iterators
static ref COMPLETING_METHODS: [(Symbol, usize); 12] = [
    (*sym::count, 1),
    (*sym::fold, 3),
    (*sym::for_each, 2),
    (*sym::partition, 2),
    (*sym::max, 1),
    (*sym::max_by, 2),
    (*sym::max_by_key, 2),
    (*sym::min, 1),
    (*sym::min_by, 2),
    (*sym::min_by_key, 2),
    (*sym::sum, 1),
    (*sym::product, 1),
];

/// the paths of types that are known to be infinitely allocating
static ref INFINITE_COLLECTORS: [Vec<Symbol>; 8] = [
    paths::BINARY_HEAP.to_vec(),
    paths::BTREEMAP.to_vec(),
    paths::BTREESET.to_vec(),
    paths::HASHMAP.to_vec(),
    paths::HASHSET.to_vec(),
    paths::LINKED_LIST.to_vec(),
    paths::VEC.to_vec(),
    paths::VEC_DEQUE.to_vec(),
];
}

fn complete_infinite_iter(cx: &LateContext<'_, '_>, expr: &Expr) -> Finiteness {
    match expr.node {
        ExprKind::MethodCall(ref method, _, ref args) => {
            for &(name, len) in COMPLETING_METHODS.iter() {
                if method.ident.name == name && args.len() == len {
                    return is_infinite(cx, &args[0]);
                }
            }
            for &(name, len) in POSSIBLY_COMPLETING_METHODS.iter() {
                if method.ident.name == name && args.len() == len {
                    return MaybeInfinite.and(is_infinite(cx, &args[0]));
                }
            }
            if method.ident.name == *sym::last && args.len() == 1 {
                let not_double_ended = get_trait_def_id(cx, &*paths::DOUBLE_ENDED_ITERATOR)
                    .map_or(false, |id| !implements_trait(cx, cx.tables.expr_ty(&args[0]), id, &[]));
                if not_double_ended {
                    return is_infinite(cx, &args[0]);
                }
            } else if method.ident.name == *sym::collect {
                let ty = cx.tables.expr_ty(expr);
                if INFINITE_COLLECTORS.iter().any(|path| match_type(cx, ty, path)) {
                    return is_infinite(cx, &args[0]);
                }
            }
        },
        ExprKind::Binary(op, ref l, ref r) => {
            if op.node.is_comparison() {
                return is_infinite(cx, l).and(is_infinite(cx, r)).and(MaybeInfinite);
            }
        }, // TODO: ExprKind::Loop + Match
        _ => (),
    }
    Finite
}
