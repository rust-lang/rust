use rustc::lint::*;
use rustc::middle::const_eval::ConstVal::{Int, Uint};
use rustc::middle::const_eval::EvalHint::ExprTypeChecked;
use rustc::middle::const_eval::{eval_const_expr_partial, ConstVal};
use rustc::middle::ty;
use rustc_front::hir::*;
use std::cmp::Ordering;
use syntax::ast::Lit_::LitBool;
use syntax::codemap::Span;

use utils::{snippet, span_lint, span_note_and_lint, span_help_and_lint, in_external_macro, expr_block};

/// **What it does:** This lint checks for matches with a single arm where an `if let` will usually suffice. It is `Warn` by default.
///
/// **Why is this bad?** Just readability – `if let` nests less than a `match`.
///
/// **Known problems:** None
///
/// **Example:**
/// ```
/// match x {
///     Some(ref foo) -> bar(foo),
///     _ => ()
/// }
/// ```
declare_lint!(pub SINGLE_MATCH, Warn,
              "a match statement with a single nontrivial arm (i.e, where the other arm \
               is `_ => {}`) is used; recommends `if let` instead");

/// **What it does:** This lint checks for matches where all arms match a reference, suggesting to remove the reference and deref the matched expression instead. It is `Warn` by default.
///
/// **Why is this bad?** It just makes the code less readable. That reference destructuring adds nothing to the code.
///
/// **Known problems:** None
///
/// **Example:**
///
/// ```
/// match x {
///     &A(ref y) => foo(y),
///     &B => bar(),
///     _ => frob(&x),
/// }
/// ```
declare_lint!(pub MATCH_REF_PATS, Warn,
              "a match has all arms prefixed with `&`; the match expression can be \
               dereferenced instead");

/// **What it does:** This lint checks for matches where match expression is a `bool`. It suggests to replace the expression with an `if...else` block. It is `Warn` by default.
///
/// **Why is this bad?** It makes the code less readable.
///
/// **Known problems:** None
///
/// **Example:**
///
/// ```
/// let condition: bool = true;
/// match condition {
///     true => foo(),
///     false => bar(),
/// }
/// ```
declare_lint!(pub MATCH_BOOL, Warn,
              "a match on boolean expression; recommends `if..else` block instead");

/// **What it does:** This lint checks for overlapping match arms. It is `Warn` by default.
///
/// **Why is this bad?** It is likely to be an error and if not, makes the code less obvious.
///
/// **Known problems:** None
///
/// **Example:**
///
/// ```
/// let x = 5;
/// match x {
///     1 ... 10 => println!("1 ... 10"),
///     5 ... 15 => println!("5 ... 15"),
///     _ => (),
/// }
/// ```
declare_lint!(pub MATCH_OVERLAPPING_ARM, Warn,
              "overlapping match arms");

#[allow(missing_copy_implementations)]
pub struct MatchPass;

impl LintPass for MatchPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(SINGLE_MATCH, MATCH_REF_PATS, MATCH_BOOL)
    }
}

impl LateLintPass for MatchPass {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if in_external_macro(cx, expr.span) { return; }
        if let ExprMatch(ref ex, ref arms, MatchSource::Normal) = expr.node {
            // check preconditions for SINGLE_MATCH
                // only two arms
            if arms.len() == 2 &&
                // both of the arms have a single pattern and no guard
                arms[0].pats.len() == 1 && arms[0].guard.is_none() &&
                arms[1].pats.len() == 1 && arms[1].guard.is_none() &&
                // and the second pattern is a `_` wildcard: this is not strictly necessary,
                // since the exhaustiveness check will ensure the last one is a catch-all,
                // but in some cases, an explicit match is preferred to catch situations
                // when an enum is extended, so we don't consider these cases
                arms[1].pats[0].node == PatWild &&
                // we don't want any content in the second arm (unit or empty block)
                is_unit_expr(&arms[1].body) &&
                // finally, MATCH_BOOL doesn't apply here
                (cx.tcx.expr_ty(ex).sty != ty::TyBool || cx.current_level(MATCH_BOOL) == Allow)
            {
                span_help_and_lint(cx, SINGLE_MATCH, expr.span,
                                   "you seem to be trying to use match for destructuring a \
                                    single pattern. Consider using `if let`",
                                   &format!("try\nif let {} = {} {}",
                                            snippet(cx, arms[0].pats[0].span, ".."),
                                            snippet(cx, ex.span, ".."),
                                            expr_block(cx, &arms[0].body, None, "..")));
            }

            // check preconditions for MATCH_BOOL
            // type of expression == bool
            if cx.tcx.expr_ty(ex).sty == ty::TyBool {
                if arms.len() == 2 && arms[0].pats.len() == 1 { // no guards
                    let exprs = if let PatLit(ref arm_bool) = arms[0].pats[0].node {
                        if let ExprLit(ref lit) = arm_bool.node {
                            match lit.node {
                                LitBool(true) => Some((&*arms[0].body, &*arms[1].body)),
                                LitBool(false) => Some((&*arms[1].body, &*arms[0].body)),
                                _ => None,
                            }
                        } else { None }
                    } else { None };
                    if let Some((ref true_expr, ref false_expr)) = exprs {
                        if !is_unit_expr(true_expr) {
                            if !is_unit_expr(false_expr) {
                                span_help_and_lint(cx, MATCH_BOOL, expr.span,
                                    "you seem to be trying to match on a boolean expression. \
                                   Consider using an if..else block:",
                                   &format!("try\nif {} {} else {}",
                                        snippet(cx, ex.span, "b"),
                                        expr_block(cx, true_expr, None, ".."),
                                        expr_block(cx, false_expr, None, "..")));
                            } else {
                                span_help_and_lint(cx, MATCH_BOOL, expr.span,
                                    "you seem to be trying to match on a boolean expression. \
                                   Consider using an if..else block:",
                                   &format!("try\nif {} {}",
                                        snippet(cx, ex.span, "b"),
                                        expr_block(cx, true_expr, None, "..")));
                            }
                        } else if !is_unit_expr(false_expr) {
                            span_help_and_lint(cx, MATCH_BOOL, expr.span,
                                "you seem to be trying to match on a boolean expression. \
                               Consider using an if..else block:",
                               &format!("try\nif !{} {}",
                                    snippet(cx, ex.span, "b"),
                                    expr_block(cx, false_expr, None, "..")));
                        } else {
                            span_lint(cx, MATCH_BOOL, expr.span,
                                   "you seem to be trying to match on a boolean expression. \
                                   Consider using an if..else block");
                        }
                    } else {
                        span_lint(cx, MATCH_BOOL, expr.span,
                            "you seem to be trying to match on a boolean expression. \
                            Consider using an if..else block");
                    }
                } else {
                    span_lint(cx, MATCH_BOOL, expr.span,
                        "you seem to be trying to match on a boolean expression. \
                        Consider using an if..else block");
                }
            }

            // MATCH_OVERLAPPING_ARM
            if arms.len() >= 2 {
                let ranges = all_ranges(cx, arms);
                let overlap = match type_ranges(&ranges) {
                    TypedRanges::IntRanges(ranges) => overlaping(&ranges).map(|(start, end)| (start.span, end.span)),
                    TypedRanges::UintRanges(ranges) => overlaping(&ranges).map(|(start, end)| (start.span, end.span)),
                    TypedRanges::None => None,
                };

                if let Some((start, end)) = overlap {
                    span_note_and_lint(cx, MATCH_OVERLAPPING_ARM, start,
                                       "some ranges overlap",
                                       end, "overlaps with this");
                }
            }
        }
        if let ExprMatch(ref ex, ref arms, source) = expr.node {
            // check preconditions for MATCH_REF_PATS
            if has_only_ref_pats(arms) {
                if let ExprAddrOf(Mutability::MutImmutable, ref inner) = ex.node {
                    let template = match_template(cx, expr.span, source, "", inner);
                    span_lint(cx, MATCH_REF_PATS, expr.span, &format!(
                        "you don't need to add `&` to both the expression \
                         and the patterns: use `{}`", template));
                } else {
                    let template = match_template(cx, expr.span, source, "*", ex);
                    span_lint(cx, MATCH_REF_PATS, expr.span, &format!(
                        "instead of prefixing all patterns with `&`, you can dereference the \
                         expression: `{}`", template));
                }
            }
        }
    }
}

/// Get all arms that are unbounded PatRange-s.
fn all_ranges(cx: &LateContext, arms: &[Arm]) -> Vec<SpannedRange<ConstVal>> {
    arms.iter()
        .filter_map(|arm| {
            if let Arm { ref pats, guard: None, .. } = *arm {
                Some(pats.iter().filter_map(|pat| {
                    if_let_chain! {[
                        let PatRange(ref lhs, ref rhs) = pat.node,
                        let Ok(lhs) = eval_const_expr_partial(cx.tcx, &lhs, ExprTypeChecked, None),
                        let Ok(rhs) = eval_const_expr_partial(cx.tcx, &rhs, ExprTypeChecked, None)
                    ], {
                        return Some(SpannedRange { span: pat.span, node: (lhs, rhs) });
                    }}

                    None
                }))
            }
            else {
                None
            }
        })
        .flat_map(IntoIterator::into_iter)
        .collect()
}

#[derive(Debug, Eq, PartialEq)]
struct SpannedRange<T> {
    span: Span,
    node: (T, T),
}

#[derive(Debug)]
enum TypedRanges {
    IntRanges(Vec<SpannedRange<i64>>),
    UintRanges(Vec<SpannedRange<u64>>),
    None,
}

/// Get all `Int` ranges or all `Uint` ranges. Mixed types are an error anyway and other types than
/// `Uint` and `Int` probably don't make sense.
fn type_ranges(ranges: &[SpannedRange<ConstVal>]) -> TypedRanges {
    if ranges.is_empty() {
        TypedRanges::None
    }
    else {
        match ranges[0].node {
            (Int(_), Int(_)) => {
                TypedRanges::IntRanges(ranges.iter().filter_map(|range| {
                    if let (Int(start), Int(end)) = range.node {
                        Some(SpannedRange { span: range.span, node: (start, end) })
                    }
                    else {
                        None
                    }
                }).collect())
            },
            (Uint(_), Uint(_)) => {
                TypedRanges::UintRanges(ranges.iter().filter_map(|range| {
                    if let (Uint(start), Uint(end)) = range.node {
                        Some(SpannedRange { span: range.span, node: (start, end) })
                    }
                    else {
                        None
                    }
                }).collect())
            },
            _ => TypedRanges::None,
        }
    }
}

fn is_unit_expr(expr: &Expr) -> bool {
    match expr.node {
        ExprTup(ref v) if v.is_empty() => true,
        ExprBlock(ref b) if b.stmts.is_empty() && b.expr.is_none() => true,
        _ => false,
    }
}

fn has_only_ref_pats(arms: &[Arm]) -> bool {
    let mapped = arms.iter().flat_map(|a| &a.pats).map(|p| match p.node {
        PatRegion(..) => Some(true),  // &-patterns
        PatWild => Some(false),   // an "anything" wildcard is also fine
        _ => None,                    // any other pattern is not fine
    }).collect::<Option<Vec<bool>>>();
    // look for Some(v) where there's at least one true element
    mapped.map_or(false, |v| v.iter().any(|el| *el))
}

fn match_template(cx: &LateContext,
                  span: Span,
                  source: MatchSource,
                  op: &str,
                  expr: &Expr) -> String {
    let expr_snippet = snippet(cx, expr.span, "..");
    match source {
        MatchSource::Normal => {
            format!("match {}{} {{ ...", op, expr_snippet)
        }
        MatchSource::IfLetDesugar { .. } => {
            format!("if let ... = {}{} {{", op, expr_snippet)
        }
        MatchSource::WhileLetDesugar => {
            format!("while let ... = {}{} {{", op, expr_snippet)
        }
        MatchSource::ForLoopDesugar => {
            cx.sess().span_bug(span, "for loop desugared to match with &-patterns!")
        }
    }
}

fn overlaping<T>(ranges: &[SpannedRange<T>]) -> Option<(&SpannedRange<T>, &SpannedRange<T>)>
    where T: Copy + Ord {
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    enum Kind<'a, T: 'a> {
        Start(T, &'a SpannedRange<T>),
        End(T, &'a SpannedRange<T>),
    }

    impl<'a, T: Copy> Kind<'a, T> {
        fn range(&self) -> &'a SpannedRange<T> {
            match *self {
                Kind::Start(_, r) | Kind::End(_, r) => r
            }
        }

        fn value(self) -> T {
            match self {
                Kind::Start(t, _) | Kind::End(t, _) => t
            }
        }
    }

    impl<'a, T: Copy + Ord> PartialOrd for Kind<'a, T> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl<'a, T: Copy + Ord> Ord for Kind<'a, T> {
        fn cmp(&self, other: &Self) -> Ordering {
            self.value().cmp(&other.value())
        }
    }

    let mut values = Vec::with_capacity(2*ranges.len());

    for r in ranges {
        values.push(Kind::Start(r.node.0, &r));
        values.push(Kind::End(r.node.1, &r));
    }

    values.sort();

    for (a, b) in values.iter().zip(values.iter().skip(1)) {
        match (a, b) {
            (&Kind::Start(_, ra), &Kind::End(_, rb)) => if ra.node != rb.node { return Some((ra, rb)) },
            (&Kind::End(a, _), &Kind::Start(b, _)) if a != b => (),
            _ => return Some((&a.range(), &b.range())),
        }
    }

    None
}

#[test]
fn test_overlapping() {
    use syntax::codemap::DUMMY_SP;

    let sp = |s, e| SpannedRange { span: DUMMY_SP, node: (s, e) };

    assert_eq!(None, overlaping::<u8>(&[]));
    assert_eq!(None, overlaping(&[sp(1, 4)]));
    assert_eq!(None, overlaping(&[sp(1, 4), sp(5, 6)]));
    assert_eq!(None, overlaping(&[sp(1, 4), sp(5, 6), sp(10, 11)]));
    assert_eq!(Some((&sp(1, 4), &sp(3, 6))), overlaping(&[sp(1, 4), sp(3, 6)]));
    assert_eq!(Some((&sp(5, 6), &sp(6, 11))), overlaping(&[sp(1, 4), sp(5, 6), sp(6, 11)]));
}
