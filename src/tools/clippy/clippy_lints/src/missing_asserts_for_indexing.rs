use std::mem;
use std::ops::ControlFlow;

use clippy_utils::comparisons::{Rel, normalize_comparison};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::macros::{find_assert_eq_args, first_node_macro_backtrace};
use clippy_utils::source::snippet;
use clippy_utils::visitors::for_each_expr_without_closures;
use clippy_utils::{eq_expr_value, hash_expr, higher};
use rustc_ast::{BinOpKind, LitKind, RangeLimits};
use rustc_data_structures::packed::Pu128;
use rustc_data_structures::unhash::UnindexMap;
use rustc_errors::{Applicability, Diag};
use rustc_hir::{Block, Body, Expr, ExprKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::source_map::Spanned;
use rustc_span::{Span, sym};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for repeated slice indexing without asserting beforehand that the length
    /// is greater than the largest index used to index into the slice.
    ///
    /// ### Why restrict this?
    /// In the general case where the compiler does not have a lot of information
    /// about the length of a slice, indexing it repeatedly will generate a bounds check
    /// for every single index.
    ///
    /// Asserting that the length of the slice is at least as large as the largest value
    /// to index beforehand gives the compiler enough information to elide the bounds checks,
    /// effectively reducing the number of bounds checks from however many times
    /// the slice was indexed to just one (the assert).
    ///
    /// ### Drawbacks
    /// False positives. It is, in general, very difficult to predict how well
    /// the optimizer will be able to elide bounds checks and it very much depends on
    /// the surrounding code. For example, indexing into the slice yielded by the
    /// [`slice::chunks_exact`](https://doc.rust-lang.org/stable/std/primitive.slice.html#method.chunks_exact)
    /// iterator will likely have all of the bounds checks elided even without an assert
    /// if the `chunk_size` is a constant.
    ///
    /// Asserts are not tracked across function calls. Asserting the length of a slice
    /// in a different function likely gives the optimizer enough information
    /// about the length of a slice, but this lint will not detect that.
    ///
    /// ### Example
    /// ```no_run
    /// fn sum(v: &[u8]) -> u8 {
    ///     // 4 bounds checks
    ///     v[0] + v[1] + v[2] + v[3]
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn sum(v: &[u8]) -> u8 {
    ///     assert!(v.len() > 3);
    ///     // no bounds checks
    ///     v[0] + v[1] + v[2] + v[3]
    /// }
    /// ```
    #[clippy::version = "1.74.0"]
    pub MISSING_ASSERTS_FOR_INDEXING,
    restriction,
    "indexing into a slice multiple times without an `assert`"
}
declare_lint_pass!(MissingAssertsForIndexing => [MISSING_ASSERTS_FOR_INDEXING]);

fn report_lint<F>(cx: &LateContext<'_>, full_span: Span, msg: &'static str, indexes: &[Span], f: F)
where
    F: FnOnce(&mut Diag<'_, ()>),
{
    span_lint_and_then(cx, MISSING_ASSERTS_FOR_INDEXING, full_span, msg, |diag| {
        f(diag);
        for span in indexes {
            diag.span_note(*span, "slice indexed here");
        }
        diag.note("asserting the length before indexing will elide bounds checks");
    });
}

#[derive(Copy, Clone, Debug)]
enum LengthComparison {
    /// `v.len() < 5`
    LengthLessThanInt,
    /// `5 < v.len()`
    IntLessThanLength,
    /// `v.len() <= 5`
    LengthLessThanOrEqualInt,
    /// `5 <= v.len()`
    IntLessThanOrEqualLength,
    /// `5 == v.len()`
    /// `v.len() == 5`
    LengthEqualInt,
}

/// Extracts parts out of a length comparison expression.
///
/// E.g. for `v.len() > 5` this returns `Some((LengthComparison::IntLessThanLength, 5, v.len()))`
fn len_comparison<'hir>(
    bin_op: BinOpKind,
    left: &'hir Expr<'hir>,
    right: &'hir Expr<'hir>,
) -> Option<(LengthComparison, usize, &'hir Expr<'hir>)> {
    macro_rules! int_lit_pat {
        ($id:ident) => {
            ExprKind::Lit(&Spanned {
                node: LitKind::Int(Pu128($id), _),
                ..
            })
        };
    }

    // normalize comparison, `v.len() > 4` becomes `4 < v.len()`
    // this simplifies the logic a bit
    let (op, left, right) = normalize_comparison(bin_op, left, right)?;
    match (op, left.kind, right.kind) {
        (Rel::Lt, int_lit_pat!(left), _) => Some((LengthComparison::IntLessThanLength, left as usize, right)),
        (Rel::Lt, _, int_lit_pat!(right)) => Some((LengthComparison::LengthLessThanInt, right as usize, left)),
        (Rel::Le, int_lit_pat!(left), _) => Some((LengthComparison::IntLessThanOrEqualLength, left as usize, right)),
        (Rel::Le, _, int_lit_pat!(right)) => Some((LengthComparison::LengthLessThanOrEqualInt, right as usize, left)),
        (Rel::Eq, int_lit_pat!(left), _) => Some((LengthComparison::LengthEqualInt, left as usize, right)),
        (Rel::Eq, _, int_lit_pat!(right)) => Some((LengthComparison::LengthEqualInt, right as usize, left)),
        _ => None,
    }
}

/// Attempts to extract parts out of an `assert!`-like expression
/// in the form `assert!(some_slice.len() > 5)`.
///
/// `assert!` has expanded to an if expression at the HIR, so this
/// actually works not just with `assert!` specifically, but anything
/// that has a never type expression in the `then` block (e.g. `panic!`).
fn assert_len_expr<'hir>(
    cx: &LateContext<'_>,
    expr: &'hir Expr<'hir>,
) -> Option<(LengthComparison, usize, &'hir Expr<'hir>)> {
    let (cmp, asserted_len, slice_len) = if let Some(higher::If { cond, then, .. }) = higher::If::hir(expr)
        && let ExprKind::Unary(UnOp::Not, condition) = &cond.kind
        && let ExprKind::Binary(bin_op, left, right) = &condition.kind
        // check if `then` block has a never type expression
        && let ExprKind::Block(Block { expr: Some(then_expr), .. }, _) = then.kind
        && cx.typeck_results().expr_ty(then_expr).is_never()
    {
        len_comparison(bin_op.node, left, right)?
    } else if let Some((macro_call, bin_op)) = first_node_macro_backtrace(cx, expr).find_map(|macro_call| {
        match cx.tcx.get_diagnostic_name(macro_call.def_id) {
            Some(sym::assert_eq_macro) => Some((macro_call, BinOpKind::Eq)),
            Some(sym::assert_ne_macro) => Some((macro_call, BinOpKind::Ne)),
            _ => None,
        }
    }) && let Some((left, right, _)) = find_assert_eq_args(cx, expr, macro_call.expn)
    {
        len_comparison(bin_op, left, right)?
    } else {
        return None;
    };

    if let ExprKind::MethodCall(method, recv, [], _) = &slice_len.kind
        && cx.typeck_results().expr_ty_adjusted(recv).peel_refs().is_slice()
        && method.ident.name == sym::len
    {
        Some((cmp, asserted_len, recv))
    } else {
        None
    }
}

#[derive(Debug)]
enum IndexEntry<'hir> {
    /// `assert!` without any indexing (so far)
    StrayAssert {
        asserted_len: usize,
        comparison: LengthComparison,
        assert_span: Span,
        slice: &'hir Expr<'hir>,
    },
    /// `assert!` with indexing
    ///
    /// We also store the highest index to be able to check
    /// if the `assert!` asserts the right length.
    AssertWithIndex {
        highest_index: usize,
        is_first_highest: bool,
        asserted_len: usize,
        assert_span: Span,
        slice: &'hir Expr<'hir>,
        indexes: Vec<Span>,
        comparison: LengthComparison,
    },
    /// Indexing without an `assert!`
    IndexWithoutAssert {
        highest_index: usize,
        is_first_highest: bool,
        indexes: Vec<Span>,
        slice: &'hir Expr<'hir>,
    },
}

impl<'hir> IndexEntry<'hir> {
    pub fn slice(&self) -> &'hir Expr<'hir> {
        match self {
            IndexEntry::StrayAssert { slice, .. }
            | IndexEntry::AssertWithIndex { slice, .. }
            | IndexEntry::IndexWithoutAssert { slice, .. } => slice,
        }
    }

    pub fn index_spans(&self) -> Option<&[Span]> {
        match self {
            IndexEntry::StrayAssert { .. } => None,
            IndexEntry::AssertWithIndex { indexes, .. } | IndexEntry::IndexWithoutAssert { indexes, .. } => {
                Some(indexes)
            },
        }
    }
}

/// Extracts the upper index of a slice indexing expression.
///
/// E.g. for `5` this returns `Some(5)`, for `..5` this returns `Some(4)`,
/// for `..=5` this returns `Some(5)`
fn upper_index_expr(expr: &Expr<'_>) -> Option<usize> {
    if let ExprKind::Lit(lit) = &expr.kind
        && let LitKind::Int(Pu128(index), _) = lit.node
    {
        Some(index as usize)
    } else if let Some(higher::Range {
        end: Some(end), limits, ..
    }) = higher::Range::hir(expr)
        && let ExprKind::Lit(lit) = &end.kind
        && let LitKind::Int(Pu128(index @ 1..), _) = lit.node
    {
        match limits {
            RangeLimits::HalfOpen => Some(index as usize - 1),
            RangeLimits::Closed => Some(index as usize),
        }
    } else {
        None
    }
}

/// Checks if the expression is an index into a slice and adds it to `indexes`
fn check_index<'hir>(cx: &LateContext<'_>, expr: &'hir Expr<'hir>, map: &mut UnindexMap<u64, Vec<IndexEntry<'hir>>>) {
    if let ExprKind::Index(slice, index_lit, _) = expr.kind
        && cx.typeck_results().expr_ty_adjusted(slice).peel_refs().is_slice()
        && let Some(index) = upper_index_expr(index_lit)
    {
        let hash = hash_expr(cx, slice);

        let indexes = map.entry(hash).or_default();
        let entry = indexes.iter_mut().find(|entry| eq_expr_value(cx, entry.slice(), slice));

        if let Some(entry) = entry {
            match entry {
                IndexEntry::StrayAssert {
                    asserted_len,
                    comparison,
                    assert_span,
                    slice,
                } => {
                    if slice.span.lo() > assert_span.lo() {
                        *entry = IndexEntry::AssertWithIndex {
                            highest_index: index,
                            is_first_highest: true,
                            asserted_len: *asserted_len,
                            assert_span: *assert_span,
                            slice,
                            indexes: vec![expr.span],
                            comparison: *comparison,
                        };
                    }
                },
                IndexEntry::IndexWithoutAssert {
                    highest_index,
                    indexes,
                    is_first_highest,
                    ..
                }
                | IndexEntry::AssertWithIndex {
                    highest_index,
                    indexes,
                    is_first_highest,
                    ..
                } => {
                    indexes.push(expr.span);
                    if *is_first_highest {
                        (*is_first_highest) = *highest_index >= index;
                    }
                    *highest_index = (*highest_index).max(index);
                },
            }
        } else {
            indexes.push(IndexEntry::IndexWithoutAssert {
                highest_index: index,
                is_first_highest: true,
                indexes: vec![expr.span],
                slice,
            });
        }
    }
}

/// Checks if the expression is an `assert!` expression and adds it to `asserts`
fn check_assert<'hir>(cx: &LateContext<'_>, expr: &'hir Expr<'hir>, map: &mut UnindexMap<u64, Vec<IndexEntry<'hir>>>) {
    if let Some((comparison, asserted_len, slice)) = assert_len_expr(cx, expr) {
        let hash = hash_expr(cx, slice);
        let indexes = map.entry(hash).or_default();

        let entry = indexes.iter_mut().find(|entry| eq_expr_value(cx, entry.slice(), slice));

        if let Some(entry) = entry {
            if let IndexEntry::IndexWithoutAssert {
                highest_index,
                is_first_highest,
                indexes,
                slice,
            } = entry
                && expr.span.lo() <= slice.span.lo()
            {
                *entry = IndexEntry::AssertWithIndex {
                    highest_index: *highest_index,
                    indexes: mem::take(indexes),
                    is_first_highest: *is_first_highest,
                    slice,
                    assert_span: expr.span.source_callsite(),
                    comparison,
                    asserted_len,
                };
            }
        } else {
            indexes.push(IndexEntry::StrayAssert {
                asserted_len,
                comparison,
                assert_span: expr.span.source_callsite(),
                slice,
            });
        }
    }
}

/// Inspects indexes and reports lints.
///
/// Called at the end of this lint after all indexing and `assert!` expressions have been collected.
fn report_indexes(cx: &LateContext<'_>, map: &UnindexMap<u64, Vec<IndexEntry<'_>>>) {
    for bucket in map.values() {
        for entry in bucket {
            let Some(full_span) = entry
                .index_spans()
                .and_then(|spans| spans.first().zip(spans.last()))
                .map(|(low, &high)| low.to(high))
            else {
                continue;
            };

            match *entry {
                IndexEntry::AssertWithIndex {
                    highest_index,
                    is_first_highest,
                    asserted_len,
                    ref indexes,
                    comparison,
                    assert_span,
                    slice,
                } if indexes.len() > 1 && !is_first_highest => {
                    // if we have found an `assert!`, let's also check that it's actually right
                    // and if it covers the highest index and if not, suggest the correct length
                    let sugg = match comparison {
                        // `v.len() < 5` and `v.len() <= 5` does nothing in terms of bounds checks.
                        // The user probably meant `v.len() > 5`
                        LengthComparison::LengthLessThanInt | LengthComparison::LengthLessThanOrEqualInt => Some(
                            format!("assert!({}.len() > {highest_index})", snippet(cx, slice.span, "..")),
                        ),
                        // `5 < v.len()` == `v.len() > 5`
                        LengthComparison::IntLessThanLength if asserted_len < highest_index => Some(format!(
                            "assert!({}.len() > {highest_index})",
                            snippet(cx, slice.span, "..")
                        )),
                        // `5 <= v.len() == `v.len() >= 5`
                        LengthComparison::IntLessThanOrEqualLength if asserted_len <= highest_index => Some(format!(
                            "assert!({}.len() > {highest_index})",
                            snippet(cx, slice.span, "..")
                        )),
                        // `highest_index` here is rather a length, so we need to add 1 to it
                        LengthComparison::LengthEqualInt if asserted_len < highest_index + 1 => Some(format!(
                            "assert!({}.len() == {})",
                            snippet(cx, slice.span, ".."),
                            highest_index + 1
                        )),
                        _ => None,
                    };

                    if let Some(sugg) = sugg {
                        report_lint(
                            cx,
                            full_span,
                            "indexing into a slice multiple times with an `assert` that does not cover the highest index",
                            indexes,
                            |diag| {
                                diag.span_suggestion(
                                    assert_span,
                                    "provide the highest index that is indexed with",
                                    sugg,
                                    Applicability::MachineApplicable,
                                );
                            },
                        );
                    }
                },
                IndexEntry::IndexWithoutAssert {
                    ref indexes,
                    highest_index,
                    is_first_highest,
                    slice,
                } if indexes.len() > 1 && !is_first_highest => {
                    // if there was no `assert!` but more than one index, suggest
                    // adding an `assert!` that covers the highest index
                    report_lint(
                        cx,
                        full_span,
                        "indexing into a slice multiple times without an `assert`",
                        indexes,
                        |diag| {
                            diag.help(format!(
                                "consider asserting the length before indexing: `assert!({}.len() > {highest_index});`",
                                snippet(cx, slice.span, "..")
                            ));
                        },
                    );
                },
                _ => {},
            }
        }
    }
}

impl LateLintPass<'_> for MissingAssertsForIndexing {
    fn check_body(&mut self, cx: &LateContext<'_>, body: &Body<'_>) {
        let mut map = UnindexMap::default();

        for_each_expr_without_closures(body.value, |expr| {
            check_index(cx, expr, &mut map);
            check_assert(cx, expr, &mut map);
            ControlFlow::<!, ()>::Continue(())
        });

        report_indexes(cx, &map);
    }
}
