use clippy_utils::consts::{constant, constant_full_int, miri_to_const, FullInt};
use clippy_utils::diagnostics::span_lint_and_note;
use core::cmp::Ordering;
use rustc_hir::{Arm, Expr, PatKind, RangeEnd};
use rustc_lint::LateContext;
use rustc_middle::mir;
use rustc_middle::ty::Ty;
use rustc_span::Span;

use super::MATCH_OVERLAPPING_ARM;

pub(crate) fn check<'tcx>(cx: &LateContext<'tcx>, ex: &'tcx Expr<'_>, arms: &'tcx [Arm<'_>]) {
    if arms.len() >= 2 && cx.typeck_results().expr_ty(ex).is_integral() {
        let ranges = all_ranges(cx, arms, cx.typeck_results().expr_ty(ex));
        if !ranges.is_empty() {
            if let Some((start, end)) = overlapping(&ranges) {
                span_lint_and_note(
                    cx,
                    MATCH_OVERLAPPING_ARM,
                    start.span,
                    "some ranges overlap",
                    Some(end.span),
                    "overlaps with this",
                );
            }
        }
    }
}

/// Gets the ranges for each range pattern arm. Applies `ty` bounds for open ranges.
fn all_ranges<'tcx>(cx: &LateContext<'tcx>, arms: &'tcx [Arm<'_>], ty: Ty<'tcx>) -> Vec<SpannedRange<FullInt>> {
    arms.iter()
        .filter_map(|arm| {
            if let Arm { pat, guard: None, .. } = *arm {
                if let PatKind::Range(ref lhs, ref rhs, range_end) = pat.kind {
                    let lhs_const = match lhs {
                        Some(lhs) => constant(cx, cx.typeck_results(), lhs)?,
                        None => {
                            let min_val_const = ty.numeric_min_val(cx.tcx)?;
                            let min_constant = mir::ConstantKind::from_value(
                                cx.tcx.valtree_to_const_val((ty, min_val_const.to_valtree())),
                                ty,
                            );
                            miri_to_const(cx, min_constant)?
                        },
                    };
                    let rhs_const = match rhs {
                        Some(rhs) => constant(cx, cx.typeck_results(), rhs)?,
                        None => {
                            let max_val_const = ty.numeric_max_val(cx.tcx)?;
                            let max_constant = mir::ConstantKind::from_value(
                                cx.tcx.valtree_to_const_val((ty, max_val_const.to_valtree())),
                                ty,
                            );
                            miri_to_const(cx, max_constant)?
                        },
                    };
                    let lhs_val = lhs_const.int_value(cx, ty)?;
                    let rhs_val = rhs_const.int_value(cx, ty)?;
                    let rhs_bound = match range_end {
                        RangeEnd::Included => EndBound::Included(rhs_val),
                        RangeEnd::Excluded => EndBound::Excluded(rhs_val),
                    };
                    return Some(SpannedRange {
                        span: pat.span,
                        node: (lhs_val, rhs_bound),
                    });
                }

                if let PatKind::Lit(value) = pat.kind {
                    let value = constant_full_int(cx, cx.typeck_results(), value)?;
                    return Some(SpannedRange {
                        span: pat.span,
                        node: (value, EndBound::Included(value)),
                    });
                }
            }
            None
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EndBound<T> {
    Included(T),
    Excluded(T),
}

#[derive(Debug, Eq, PartialEq)]
struct SpannedRange<T> {
    pub span: Span,
    pub node: (T, EndBound<T>),
}

fn overlapping<T>(ranges: &[SpannedRange<T>]) -> Option<(&SpannedRange<T>, &SpannedRange<T>)>
where
    T: Copy + Ord,
{
    #[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
    enum BoundKind {
        EndExcluded,
        Start,
        EndIncluded,
    }

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    struct RangeBound<'a, T>(T, BoundKind, &'a SpannedRange<T>);

    impl<'a, T: Copy + Ord> PartialOrd for RangeBound<'a, T> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl<'a, T: Copy + Ord> Ord for RangeBound<'a, T> {
        fn cmp(&self, RangeBound(other_value, other_kind, _): &Self) -> Ordering {
            let RangeBound(self_value, self_kind, _) = *self;
            (self_value, self_kind).cmp(&(*other_value, *other_kind))
        }
    }

    let mut values = Vec::with_capacity(2 * ranges.len());

    for r @ SpannedRange { node: (start, end), .. } in ranges {
        values.push(RangeBound(*start, BoundKind::Start, r));
        values.push(match end {
            EndBound::Excluded(val) => RangeBound(*val, BoundKind::EndExcluded, r),
            EndBound::Included(val) => RangeBound(*val, BoundKind::EndIncluded, r),
        });
    }

    values.sort();

    let mut started = vec![];

    for RangeBound(_, kind, range) in values {
        match kind {
            BoundKind::Start => started.push(range),
            BoundKind::EndExcluded | BoundKind::EndIncluded => {
                let mut overlap = None;

                while let Some(last_started) = started.pop() {
                    if last_started == range {
                        break;
                    }
                    overlap = Some(last_started);
                }

                if let Some(first_overlapping) = overlap {
                    return Some((range, first_overlapping));
                }
            },
        }
    }

    None
}

#[test]
fn test_overlapping() {
    use rustc_span::source_map::DUMMY_SP;

    let sp = |s, e| SpannedRange {
        span: DUMMY_SP,
        node: (s, e),
    };

    assert_eq!(None, overlapping::<u8>(&[]));
    assert_eq!(None, overlapping(&[sp(1, EndBound::Included(4))]));
    assert_eq!(
        None,
        overlapping(&[sp(1, EndBound::Included(4)), sp(5, EndBound::Included(6))])
    );
    assert_eq!(
        None,
        overlapping(&[
            sp(1, EndBound::Included(4)),
            sp(5, EndBound::Included(6)),
            sp(10, EndBound::Included(11))
        ],)
    );
    assert_eq!(
        Some((&sp(1, EndBound::Included(4)), &sp(3, EndBound::Included(6)))),
        overlapping(&[sp(1, EndBound::Included(4)), sp(3, EndBound::Included(6))])
    );
    assert_eq!(
        Some((&sp(5, EndBound::Included(6)), &sp(6, EndBound::Included(11)))),
        overlapping(&[
            sp(1, EndBound::Included(4)),
            sp(5, EndBound::Included(6)),
            sp(6, EndBound::Included(11))
        ],)
    );
}
