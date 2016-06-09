use rustc::hir::*;
use rustc::lint::*;
use rustc::middle::const_val::ConstVal;
use rustc::ty;
use rustc_const_eval::EvalHint::ExprTypeChecked;
use rustc_const_eval::eval_const_expr_partial;
use rustc_const_math::ConstInt;
use std::cmp::Ordering;
use syntax::ast::LitKind;
use syntax::codemap::Span;
use utils::paths;
use utils::{match_type, snippet, span_note_and_lint, span_lint_and_then, in_external_macro, expr_block};

/// **What it does:** This lint checks for matches with a single arm where an `if let` will usually suffice.
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
declare_lint! {
    pub SINGLE_MATCH, Warn,
    "a match statement with a single nontrivial arm (i.e, where the other arm \
     is `_ => {}`) is used; recommends `if let` instead"
}

/// **What it does:** This lint checks for matches with a two arms where an `if let` will usually suffice.
///
/// **Why is this bad?** Just readability – `if let` nests less than a `match`.
///
/// **Known problems:** Personal style preferences may differ
///
/// **Example:**
/// ```
/// match x {
///     Some(ref foo) -> bar(foo),
///     _ => bar(other_ref),
/// }
/// ```
declare_lint! {
    pub SINGLE_MATCH_ELSE, Allow,
    "a match statement with a two arms where the second arm's pattern is a wildcard; \
     recommends `if let` instead"
}

/// **What it does:** This lint checks for matches where all arms match a reference, suggesting to remove the reference and deref the matched expression instead. It also checks for `if let &foo = bar` blocks.
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
declare_lint! {
    pub MATCH_REF_PATS, Warn,
    "a match or `if let` has all arms prefixed with `&`; the match expression can be \
     dereferenced instead"
}

/// **What it does:** This lint checks for matches where match expression is a `bool`. It suggests to replace the expression with an `if...else` block.
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
declare_lint! {
    pub MATCH_BOOL, Warn,
    "a match on boolean expression; recommends `if..else` block instead"
}

/// **What it does:** This lint checks for overlapping match arms.
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
declare_lint! {
    pub MATCH_OVERLAPPING_ARM, Warn, "a match has overlapping arms"
}

#[allow(missing_copy_implementations)]
pub struct MatchPass;

impl LintPass for MatchPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(SINGLE_MATCH, MATCH_REF_PATS, MATCH_BOOL, SINGLE_MATCH_ELSE)
    }
}

impl LateLintPass for MatchPass {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if in_external_macro(cx, expr.span) {
            return;
        }
        if let ExprMatch(ref ex, ref arms, MatchSource::Normal) = expr.node {
            check_single_match(cx, ex, arms, expr);
            check_match_bool(cx, ex, arms, expr);
            check_overlapping_arms(cx, ex, arms);
        }
        if let ExprMatch(ref ex, ref arms, source) = expr.node {
            check_match_ref_pats(cx, ex, arms, source, expr);
        }
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
fn check_single_match(cx: &LateContext, ex: &Expr, arms: &[Arm], expr: &Expr) {
    if arms.len() == 2 &&
      arms[0].pats.len() == 1 && arms[0].guard.is_none() &&
      arms[1].pats.len() == 1 && arms[1].guard.is_none() {
        let els = if is_unit_expr(&arms[1].body) {
            None
        } else if let ExprBlock(_) = arms[1].body.node {
            // matches with blocks that contain statements are prettier as `if let + else`
            Some(&*arms[1].body)
        } else {
            // allow match arms with just expressions
            return;
        };
        let ty = cx.tcx.expr_ty(ex);
        if ty.sty != ty::TyBool || cx.current_level(MATCH_BOOL) == Allow {
            check_single_match_single_pattern(cx, ex, arms, expr, els);
            check_single_match_opt_like(cx, ex, arms, expr, ty, els);
        }
    }
}

fn check_single_match_single_pattern(cx: &LateContext, ex: &Expr, arms: &[Arm], expr: &Expr, els: Option<&Expr>) {
    if arms[1].pats[0].node == PatKind::Wild {
        let lint = if els.is_some() {
            SINGLE_MATCH_ELSE
        } else {
            SINGLE_MATCH
        };
        let els_str = els.map_or(String::new(), |els| format!(" else {}", expr_block(cx, els, None, "..")));
        span_lint_and_then(cx,
                           lint,
                           expr.span,
                           "you seem to be trying to use match for destructuring a single pattern. \
                           Consider using `if let`",
                           |db| {
            db.span_suggestion(expr.span,
                               "try this",
                               format!("if let {} = {} {}{}",
                                       snippet(cx, arms[0].pats[0].span, ".."),
                                       snippet(cx, ex.span, ".."),
                                       expr_block(cx, &arms[0].body, None, ".."),
                                       els_str));
        });
    }
}

fn check_single_match_opt_like(cx: &LateContext, ex: &Expr, arms: &[Arm], expr: &Expr, ty: ty::Ty, els: Option<&Expr>) {
    // list of candidate Enums we know will never get any more members
    let candidates = &[(&paths::COW, "Borrowed"),
                       (&paths::COW, "Cow::Borrowed"),
                       (&paths::COW, "Cow::Owned"),
                       (&paths::COW, "Owned"),
                       (&paths::OPTION, "None"),
                       (&paths::RESULT, "Err"),
                       (&paths::RESULT, "Ok")];

    let path = match arms[1].pats[0].node {
        PatKind::TupleStruct(ref path, ref inner, _) => {
            // contains any non wildcard patterns? e.g. Err(err)
            if inner.iter().any(|pat| pat.node != PatKind::Wild) {
                return;
            }
            path.to_string()
        }
        PatKind::Binding(BindByValue(MutImmutable), ident, None) => ident.node.to_string(),
        PatKind::Path(ref path) => path.to_string(),
        _ => return,
    };

    for &(ty_path, pat_path) in candidates {
        if &path == pat_path && match_type(cx, ty, ty_path) {
            let lint = if els.is_some() {
                SINGLE_MATCH_ELSE
            } else {
                SINGLE_MATCH
            };
            let els_str = els.map_or(String::new(), |els| format!(" else {}", expr_block(cx, els, None, "..")));
            span_lint_and_then(cx,
                               lint,
                               expr.span,
                               "you seem to be trying to use match for destructuring a single pattern. Consider \
                                using `if let`",
                               |db| {
                db.span_suggestion(expr.span,
                                   "try this",
                                   format!("if let {} = {} {}{}",
                                           snippet(cx, arms[0].pats[0].span, ".."),
                                           snippet(cx, ex.span, ".."),
                                           expr_block(cx, &arms[0].body, None, ".."),
                                           els_str));
            });
        }
    }
}

fn check_match_bool(cx: &LateContext, ex: &Expr, arms: &[Arm], expr: &Expr) {
    // type of expression == bool
    if cx.tcx.expr_ty(ex).sty == ty::TyBool {
        let sugg = if arms.len() == 2 && arms[0].pats.len() == 1 {
            // no guards
            let exprs = if let PatKind::Lit(ref arm_bool) = arms[0].pats[0].node {
                if let ExprLit(ref lit) = arm_bool.node {
                    match lit.node {
                        LitKind::Bool(true) => Some((&*arms[0].body, &*arms[1].body)),
                        LitKind::Bool(false) => Some((&*arms[1].body, &*arms[0].body)),
                        _ => None,
                    }
                } else {
                    None
                }
            } else {
                None
            };

            if let Some((ref true_expr, ref false_expr)) = exprs {
                match (is_unit_expr(true_expr), is_unit_expr(false_expr)) {
                    (false, false) => {
                        Some(format!("if {} {} else {}",
                                     snippet(cx, ex.span, "b"),
                                     expr_block(cx, true_expr, None, ".."),
                                     expr_block(cx, false_expr, None, "..")))
                    }
                    (false, true) => {
                        Some(format!("if {} {}", snippet(cx, ex.span, "b"), expr_block(cx, true_expr, None, "..")))
                    }
                    (true, false) => {
                        Some(format!("try\nif !{} {}",
                                     snippet(cx, ex.span, "b"),
                                     expr_block(cx, false_expr, None, "..")))
                    }
                    (true, true) => None,
                }
            } else {
                None
            }
        } else {
            None
        };

        span_lint_and_then(cx,
                           MATCH_BOOL,
                           expr.span,
                           "you seem to be trying to match on a boolean expression. Consider using an if..else block:",
                           move |db| {
                               if let Some(sugg) = sugg {
                                   db.span_suggestion(expr.span, "try this", sugg);
                               }
                           });
    }
}

fn check_overlapping_arms(cx: &LateContext, ex: &Expr, arms: &[Arm]) {
    if arms.len() >= 2 && cx.tcx.expr_ty(ex).is_integral() {
        let ranges = all_ranges(cx, arms);
        let type_ranges = type_ranges(&ranges);
        if !type_ranges.is_empty() {
            if let Some((start, end)) = overlapping(&type_ranges) {
                span_note_and_lint(cx,
                                   MATCH_OVERLAPPING_ARM,
                                   start.span,
                                   "some ranges overlap",
                                   end.span,
                                   "overlaps with this");
            }
        }
    }
}

fn check_match_ref_pats(cx: &LateContext, ex: &Expr, arms: &[Arm], source: MatchSource, expr: &Expr) {
    if has_only_ref_pats(arms) {
        if let ExprAddrOf(Mutability::MutImmutable, ref inner) = ex.node {
            let template = match_template(cx, expr.span, source, "", inner);
            span_lint_and_then(cx,
                               MATCH_REF_PATS,
                               expr.span,
                               "you don't need to add `&` to both the expression and the patterns",
                               |db| {
                                   db.span_suggestion(expr.span, "try", template);
                               });
        } else {
            let template = match_template(cx, expr.span, source, "*", ex);
            span_lint_and_then(cx,
                               MATCH_REF_PATS,
                               expr.span,
                               "you don't need to add `&` to all patterns",
                               |db| {
                                   db.span_suggestion(expr.span,
                                                      "instead of prefixing all patterns with `&`, you can \
                                                       dereference the expression",
                                                      template);
                               });
        }
    }
}

/// Get all arms that are unbounded `PatRange`s.
fn all_ranges(cx: &LateContext, arms: &[Arm]) -> Vec<SpannedRange<ConstVal>> {
    arms.iter()
        .filter_map(|arm| {
            if let Arm { ref pats, guard: None, .. } = *arm {
                Some(pats.iter().filter_map(|pat| {
                    if_let_chain! {[
                        let PatKind::Range(ref lhs, ref rhs) = pat.node,
                        let Ok(lhs) = eval_const_expr_partial(cx.tcx, lhs, ExprTypeChecked, None),
                        let Ok(rhs) = eval_const_expr_partial(cx.tcx, rhs, ExprTypeChecked, None)
                    ], {
                        return Some(SpannedRange { span: pat.span, node: (lhs, rhs) });
                    }}

                    if_let_chain! {[
                        let PatKind::Lit(ref value) = pat.node,
                        let Ok(value) = eval_const_expr_partial(cx.tcx, value, ExprTypeChecked, None)
                    ], {
                        return Some(SpannedRange { span: pat.span, node: (value.clone(), value) });
                    }}

                    None
                }))
            } else {
                None
            }
        })
        .flat_map(IntoIterator::into_iter)
        .collect()
}

#[derive(Debug, Eq, PartialEq)]
pub struct SpannedRange<T> {
    pub span: Span,
    pub node: (T, T),
}

type TypedRanges = Vec<SpannedRange<ConstInt>>;

/// Get all `Int` ranges or all `Uint` ranges. Mixed types are an error anyway and other types than
/// `Uint` and `Int` probably don't make sense.
fn type_ranges(ranges: &[SpannedRange<ConstVal>]) -> TypedRanges {
    ranges.iter()
          .filter_map(|range| {
              if let (ConstVal::Integral(start), ConstVal::Integral(end)) = range.node {
                  Some(SpannedRange {
                      span: range.span,
                      node: (start, end),
                  })
              } else {
                  None
              }
          })
          .collect()
}

fn is_unit_expr(expr: &Expr) -> bool {
    match expr.node {
        ExprTup(ref v) if v.is_empty() => true,
        ExprBlock(ref b) if b.stmts.is_empty() && b.expr.is_none() => true,
        _ => false,
    }
}

fn has_only_ref_pats(arms: &[Arm]) -> bool {
    let mapped = arms.iter()
                     .flat_map(|a| &a.pats)
                     .map(|p| {
                         match p.node {
                             PatKind::Ref(..) => Some(true),  // &-patterns
                             PatKind::Wild => Some(false),   // an "anything" wildcard is also fine
                             _ => None,                    // any other pattern is not fine
                         }
                     })
                     .collect::<Option<Vec<bool>>>();
    // look for Some(v) where there's at least one true element
    mapped.map_or(false, |v| v.iter().any(|el| *el))
}

fn match_template(cx: &LateContext, span: Span, source: MatchSource, op: &str, expr: &Expr) -> String {
    let expr_snippet = snippet(cx, expr.span, "..");
    match source {
        MatchSource::Normal => format!("match {}{} {{ .. }}", op, expr_snippet),
        MatchSource::IfLetDesugar { .. } => format!("if let .. = {}{} {{ .. }}", op, expr_snippet),
        MatchSource::WhileLetDesugar => format!("while let .. = {}{} {{ .. }}", op, expr_snippet),
        MatchSource::ForLoopDesugar => span_bug!(span, "for loop desugared to match with &-patterns!"),
        MatchSource::TryDesugar => span_bug!(span, "`?` operator desugared to match with &-patterns!"),
    }
}

pub fn overlapping<T>(ranges: &[SpannedRange<T>]) -> Option<(&SpannedRange<T>, &SpannedRange<T>)>
    where T: Copy + Ord
{
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    enum Kind<'a, T: 'a> {
        Start(T, &'a SpannedRange<T>),
        End(T, &'a SpannedRange<T>),
    }

    impl<'a, T: Copy> Kind<'a, T> {
        fn range(&self) -> &'a SpannedRange<T> {
            match *self {
                Kind::Start(_, r) |
                Kind::End(_, r) => r,
            }
        }

        fn value(self) -> T {
            match self {
                Kind::Start(t, _) |
                Kind::End(t, _) => t,
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

    let mut values = Vec::with_capacity(2 * ranges.len());

    for r in ranges {
        values.push(Kind::Start(r.node.0, r));
        values.push(Kind::End(r.node.1, r));
    }

    values.sort();

    for (a, b) in values.iter().zip(values.iter().skip(1)) {
        match (a, b) {
            (&Kind::Start(_, ra), &Kind::End(_, rb)) => {
                if ra.node != rb.node {
                    return Some((ra, rb));
                }
            }
            (&Kind::End(a, _), &Kind::Start(b, _)) if a != b => (),
            _ => return Some((a.range(), b.range())),
        }
    }

    None
}
