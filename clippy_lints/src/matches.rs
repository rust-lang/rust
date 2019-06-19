use crate::consts::{constant, Constant};
use crate::utils::paths;
use crate::utils::sugg::Sugg;
use crate::utils::{
    expr_block, in_macro_or_desugar, is_allowed, is_expn_of, match_qpath, match_type, multispan_sugg, remove_blocks,
    snippet, snippet_with_applicability, span_lint_and_sugg, span_lint_and_then, span_note_and_lint, walk_ptrs_ty,
};
use if_chain::if_chain;
use rustc::hir::def::CtorKind;
use rustc::hir::*;
use rustc::lint::{in_external_macro, LateContext, LateLintPass, LintArray, LintContext, LintPass};
use rustc::ty::{self, Ty};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;
use std::cmp::Ordering;
use std::collections::Bound;
use std::ops::Deref;
use syntax::ast::LitKind;
use syntax::source_map::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for matches with a single arm where an `if let`
    /// will usually suffice.
    ///
    /// **Why is this bad?** Just readability – `if let` nests less than a `match`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # fn bar(stool: &str) {}
    /// # let x = Some("abc");
    /// match x {
    ///     Some(ref foo) => bar(foo),
    ///     _ => (),
    /// }
    /// ```
    pub SINGLE_MATCH,
    style,
    "a match statement with a single nontrivial arm (i.e., where the other arm is `_ => {}`) instead of `if let`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for matches with a two arms where an `if let else` will
    /// usually suffice.
    ///
    /// **Why is this bad?** Just readability – `if let` nests less than a `match`.
    ///
    /// **Known problems:** Personal style preferences may differ.
    ///
    /// **Example:**
    ///
    /// Using `match`:
    ///
    /// ```rust
    /// match x {
    ///     Some(ref foo) => bar(foo),
    ///     _ => bar(other_ref),
    /// }
    /// ```
    ///
    /// Using `if let` with `else`:
    ///
    /// ```rust
    /// if let Some(ref foo) = x {
    ///     bar(foo);
    /// } else {
    ///     bar(other_ref);
    /// }
    /// ```
    pub SINGLE_MATCH_ELSE,
    pedantic,
    "a match statement with a two arms where the second arm's pattern is a placeholder instead of a specific match pattern"
}

declare_clippy_lint! {
    /// **What it does:** Checks for matches where all arms match a reference,
    /// suggesting to remove the reference and deref the matched expression
    /// instead. It also checks for `if let &foo = bar` blocks.
    ///
    /// **Why is this bad?** It just makes the code less readable. That reference
    /// destructuring adds nothing to the code.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// match x {
    ///     &A(ref y) => foo(y),
    ///     &B => bar(),
    ///     _ => frob(&x),
    /// }
    /// ```
    pub MATCH_REF_PATS,
    style,
    "a match or `if let` with all arms prefixed with `&` instead of deref-ing the match expression"
}

declare_clippy_lint! {
    /// **What it does:** Checks for matches where match expression is a `bool`. It
    /// suggests to replace the expression with an `if...else` block.
    ///
    /// **Why is this bad?** It makes the code less readable.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # fn foo() {}
    /// # fn bar() {}
    /// let condition: bool = true;
    /// match condition {
    ///     true => foo(),
    ///     false => bar(),
    /// }
    /// ```
    /// Use if/else instead:
    /// ```rust
    /// # fn foo() {}
    /// # fn bar() {}
    /// let condition: bool = true;
    /// if condition {
    ///     foo();
    /// } else {
    ///     bar();
    /// }
    /// ```
    pub MATCH_BOOL,
    style,
    "a match on a boolean expression instead of an `if..else` block"
}

declare_clippy_lint! {
    /// **What it does:** Checks for overlapping match arms.
    ///
    /// **Why is this bad?** It is likely to be an error and if not, makes the code
    /// less obvious.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let x = 5;
    /// match x {
    ///     1...10 => println!("1 ... 10"),
    ///     5...15 => println!("5 ... 15"),
    ///     _ => (),
    /// }
    /// ```
    pub MATCH_OVERLAPPING_ARM,
    style,
    "a match with overlapping arms"
}

declare_clippy_lint! {
    /// **What it does:** Checks for arm which matches all errors with `Err(_)`
    /// and take drastic actions like `panic!`.
    ///
    /// **Why is this bad?** It is generally a bad practice, just like
    /// catching all exceptions in java with `catch(Exception)`
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let x: Result<i32, &str> = Ok(3);
    /// match x {
    ///     Ok(_) => println!("ok"),
    ///     Err(_) => panic!("err"),
    /// }
    /// ```
    pub MATCH_WILD_ERR_ARM,
    style,
    "a match with `Err(_)` arm and take drastic actions"
}

declare_clippy_lint! {
    /// **What it does:** Checks for match which is used to add a reference to an
    /// `Option` value.
    ///
    /// **Why is this bad?** Using `as_ref()` or `as_mut()` instead is shorter.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let x: Option<()> = None;
    /// let r: Option<&()> = match x {
    ///     None => None,
    ///     Some(ref v) => Some(v),
    /// };
    /// ```
    pub MATCH_AS_REF,
    complexity,
    "a match on an Option value instead of using `as_ref()` or `as_mut`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for wildcard enum matches using `_`.
    ///
    /// **Why is this bad?** New enum variants added by library updates can be missed.
    ///
    /// **Known problems:** Suggested replacements may be incorrect if guards exhaustively cover some
    /// variants, and also may not use correct path to enum if it's not present in the current scope.
    ///
    /// **Example:**
    /// ```rust
    /// match x {
    ///     A => {},
    ///     _ => {},
    /// }
    /// ```
    pub WILDCARD_ENUM_MATCH_ARM,
    restriction,
    "a wildcard enum match arm using `_`"
}

declare_lint_pass!(Matches => [
    SINGLE_MATCH,
    MATCH_REF_PATS,
    MATCH_BOOL,
    SINGLE_MATCH_ELSE,
    MATCH_OVERLAPPING_ARM,
    MATCH_WILD_ERR_ARM,
    MATCH_AS_REF,
    WILDCARD_ENUM_MATCH_ARM
]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Matches {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if in_external_macro(cx.sess(), expr.span) {
            return;
        }
        if let ExprKind::Match(ref ex, ref arms, MatchSource::Normal) = expr.node {
            check_single_match(cx, ex, arms, expr);
            check_match_bool(cx, ex, arms, expr);
            check_overlapping_arms(cx, ex, arms);
            check_wild_err_arm(cx, ex, arms);
            check_wild_enum_match(cx, ex, arms);
            check_match_as_ref(cx, ex, arms, expr);
        }
        if let ExprKind::Match(ref ex, ref arms, _) = expr.node {
            check_match_ref_pats(cx, ex, arms, expr);
        }
    }
}

#[rustfmt::skip]
fn check_single_match(cx: &LateContext<'_, '_>, ex: &Expr, arms: &[Arm], expr: &Expr) {
    if arms.len() == 2 &&
      arms[0].pats.len() == 1 && arms[0].guard.is_none() &&
      arms[1].pats.len() == 1 && arms[1].guard.is_none() {
        let els = remove_blocks(&arms[1].body);
        let els = if is_unit_expr(els) {
            None
        } else if let ExprKind::Block(_, _) = els.node {
            // matches with blocks that contain statements are prettier as `if let + else`
            Some(els)
        } else {
            // allow match arms with just expressions
            return;
        };
        let ty = cx.tables.expr_ty(ex);
        if ty.sty != ty::Bool || is_allowed(cx, MATCH_BOOL, ex.hir_id) {
            check_single_match_single_pattern(cx, ex, arms, expr, els);
            check_single_match_opt_like(cx, ex, arms, expr, ty, els);
        }
    }
}

fn check_single_match_single_pattern(
    cx: &LateContext<'_, '_>,
    ex: &Expr,
    arms: &[Arm],
    expr: &Expr,
    els: Option<&Expr>,
) {
    if is_wild(&arms[1].pats[0]) {
        report_single_match_single_pattern(cx, ex, arms, expr, els);
    }
}

fn report_single_match_single_pattern(
    cx: &LateContext<'_, '_>,
    ex: &Expr,
    arms: &[Arm],
    expr: &Expr,
    els: Option<&Expr>,
) {
    let lint = if els.is_some() { SINGLE_MATCH_ELSE } else { SINGLE_MATCH };
    let els_str = els.map_or(String::new(), |els| {
        format!(" else {}", expr_block(cx, els, None, ".."))
    });
    span_lint_and_sugg(
        cx,
        lint,
        expr.span,
        "you seem to be trying to use match for destructuring a single pattern. Consider using `if \
         let`",
        "try this",
        format!(
            "if let {} = {} {}{}",
            snippet(cx, arms[0].pats[0].span, ".."),
            snippet(cx, ex.span, ".."),
            expr_block(cx, &arms[0].body, None, ".."),
            els_str,
        ),
        Applicability::HasPlaceholders,
    );
}

fn check_single_match_opt_like(
    cx: &LateContext<'_, '_>,
    ex: &Expr,
    arms: &[Arm],
    expr: &Expr,
    ty: Ty<'_>,
    els: Option<&Expr>,
) {
    // list of candidate `Enum`s we know will never get any more members
    let candidates = &[
        (&paths::COW, "Borrowed"),
        (&paths::COW, "Cow::Borrowed"),
        (&paths::COW, "Cow::Owned"),
        (&paths::COW, "Owned"),
        (&paths::OPTION, "None"),
        (&paths::RESULT, "Err"),
        (&paths::RESULT, "Ok"),
    ];

    let path = match arms[1].pats[0].node {
        PatKind::TupleStruct(ref path, ref inner, _) => {
            // Contains any non wildcard patterns (e.g., `Err(err)`)?
            if !inner.iter().all(is_wild) {
                return;
            }
            print::to_string(print::NO_ANN, |s| s.print_qpath(path, false))
        },
        PatKind::Binding(BindingAnnotation::Unannotated, .., ident, None) => ident.to_string(),
        PatKind::Path(ref path) => print::to_string(print::NO_ANN, |s| s.print_qpath(path, false)),
        _ => return,
    };

    for &(ty_path, pat_path) in candidates {
        if path == *pat_path && match_type(cx, ty, ty_path) {
            report_single_match_single_pattern(cx, ex, arms, expr, els);
        }
    }
}

fn check_match_bool(cx: &LateContext<'_, '_>, ex: &Expr, arms: &[Arm], expr: &Expr) {
    // Type of expression is `bool`.
    if cx.tables.expr_ty(ex).sty == ty::Bool {
        span_lint_and_then(
            cx,
            MATCH_BOOL,
            expr.span,
            "you seem to be trying to match on a boolean expression",
            move |db| {
                if arms.len() == 2 && arms[0].pats.len() == 1 {
                    // no guards
                    let exprs = if let PatKind::Lit(ref arm_bool) = arms[0].pats[0].node {
                        if let ExprKind::Lit(ref lit) = arm_bool.node {
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

                    if let Some((true_expr, false_expr)) = exprs {
                        let sugg = match (is_unit_expr(true_expr), is_unit_expr(false_expr)) {
                            (false, false) => Some(format!(
                                "if {} {} else {}",
                                snippet(cx, ex.span, "b"),
                                expr_block(cx, true_expr, None, ".."),
                                expr_block(cx, false_expr, None, "..")
                            )),
                            (false, true) => Some(format!(
                                "if {} {}",
                                snippet(cx, ex.span, "b"),
                                expr_block(cx, true_expr, None, "..")
                            )),
                            (true, false) => {
                                let test = Sugg::hir(cx, ex, "..");
                                Some(format!("if {} {}", !test, expr_block(cx, false_expr, None, "..")))
                            },
                            (true, true) => None,
                        };

                        if let Some(sugg) = sugg {
                            db.span_suggestion(
                                expr.span,
                                "consider using an if/else expression",
                                sugg,
                                Applicability::HasPlaceholders,
                            );
                        }
                    }
                }
            },
        );
    }
}

fn check_overlapping_arms<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, ex: &'tcx Expr, arms: &'tcx [Arm]) {
    if arms.len() >= 2 && cx.tables.expr_ty(ex).is_integral() {
        let ranges = all_ranges(cx, arms);
        let type_ranges = type_ranges(&ranges);
        if !type_ranges.is_empty() {
            if let Some((start, end)) = overlapping(&type_ranges) {
                span_note_and_lint(
                    cx,
                    MATCH_OVERLAPPING_ARM,
                    start.span,
                    "some ranges overlap",
                    end.span,
                    "overlaps with this",
                );
            }
        }
    }
}

fn is_wild(pat: &impl std::ops::Deref<Target = Pat>) -> bool {
    match pat.node {
        PatKind::Wild => true,
        _ => false,
    }
}

fn check_wild_err_arm(cx: &LateContext<'_, '_>, ex: &Expr, arms: &[Arm]) {
    let ex_ty = walk_ptrs_ty(cx.tables.expr_ty(ex));
    if match_type(cx, ex_ty, &paths::RESULT) {
        for arm in arms {
            if let PatKind::TupleStruct(ref path, ref inner, _) = arm.pats[0].node {
                let path_str = print::to_string(print::NO_ANN, |s| s.print_qpath(path, false));
                if_chain! {
                    if path_str == "Err";
                    if inner.iter().any(is_wild);
                    if let ExprKind::Block(ref block, _) = arm.body.node;
                    if is_panic_block(block);
                    then {
                        // `Err(_)` arm with `panic!` found
                        span_note_and_lint(cx,
                                           MATCH_WILD_ERR_ARM,
                                           arm.pats[0].span,
                                           "Err(_) will match all errors, maybe not a good idea",
                                           arm.pats[0].span,
                                           "to remove this warning, match each error separately \
                                            or use unreachable macro");
                    }
                }
            }
        }
    }
}

fn check_wild_enum_match(cx: &LateContext<'_, '_>, ex: &Expr, arms: &[Arm]) {
    let ty = cx.tables.expr_ty(ex);
    if !ty.is_enum() {
        // If there isn't a nice closed set of possible values that can be conveniently enumerated,
        // don't complain about not enumerating the mall.
        return;
    }

    // First pass - check for violation, but don't do much book-keeping because this is hopefully
    // the uncommon case, and the book-keeping is slightly expensive.
    let mut wildcard_span = None;
    let mut wildcard_ident = None;
    for arm in arms {
        for pat in &arm.pats {
            if let PatKind::Wild = pat.node {
                wildcard_span = Some(pat.span);
            } else if let PatKind::Binding(_, _, ident, None) = pat.node {
                wildcard_span = Some(pat.span);
                wildcard_ident = Some(ident);
            }
        }
    }

    if let Some(wildcard_span) = wildcard_span {
        // Accumulate the variants which should be put in place of the wildcard because they're not
        // already covered.

        let mut missing_variants = vec![];
        if let ty::Adt(def, _) = ty.sty {
            for variant in &def.variants {
                missing_variants.push(variant);
            }
        }

        for arm in arms {
            if arm.guard.is_some() {
                // Guards mean that this case probably isn't exhaustively covered. Technically
                // this is incorrect, as we should really check whether each variant is exhaustively
                // covered by the set of guards that cover it, but that's really hard to do.
                continue;
            }
            for pat in &arm.pats {
                if let PatKind::Path(ref path) = pat.deref().node {
                    if let QPath::Resolved(_, p) = path {
                        missing_variants.retain(|e| e.ctor_def_id != Some(p.res.def_id()));
                    }
                } else if let PatKind::TupleStruct(ref path, ..) = pat.deref().node {
                    if let QPath::Resolved(_, p) = path {
                        missing_variants.retain(|e| e.ctor_def_id != Some(p.res.def_id()));
                    }
                }
            }
        }

        let suggestion: Vec<String> = missing_variants
            .iter()
            .map(|v| {
                let suffix = match v.ctor_kind {
                    CtorKind::Fn => "(..)",
                    CtorKind::Const | CtorKind::Fictive => "",
                };
                let ident_str = if let Some(ident) = wildcard_ident {
                    format!("{} @ ", ident.name)
                } else {
                    String::new()
                };
                // This path assumes that the enum type is imported into scope.
                format!("{}{}{}", ident_str, cx.tcx.def_path_str(v.def_id), suffix)
            })
            .collect();

        if suggestion.is_empty() {
            return;
        }

        span_lint_and_sugg(
            cx,
            WILDCARD_ENUM_MATCH_ARM,
            wildcard_span,
            "wildcard match will miss any future added variants.",
            "try this",
            suggestion.join(" | "),
            Applicability::MachineApplicable,
        )
    }
}

// If the block contains only a `panic!` macro (as expression or statement)
fn is_panic_block(block: &Block) -> bool {
    match (&block.expr, block.stmts.len(), block.stmts.first()) {
        (&Some(ref exp), 0, _) => {
            is_expn_of(exp.span, "panic").is_some() && is_expn_of(exp.span, "unreachable").is_none()
        },
        (&None, 1, Some(stmt)) => {
            is_expn_of(stmt.span, "panic").is_some() && is_expn_of(stmt.span, "unreachable").is_none()
        },
        _ => false,
    }
}

fn check_match_ref_pats(cx: &LateContext<'_, '_>, ex: &Expr, arms: &[Arm], expr: &Expr) {
    if has_only_ref_pats(arms) {
        let mut suggs = Vec::new();
        let (title, msg) = if let ExprKind::AddrOf(Mutability::MutImmutable, ref inner) = ex.node {
            let span = ex.span.source_callsite();
            suggs.push((span, Sugg::hir_with_macro_callsite(cx, inner, "..").to_string()));
            (
                "you don't need to add `&` to both the expression and the patterns",
                "try",
            )
        } else {
            let span = ex.span.source_callsite();
            suggs.push((span, Sugg::hir_with_macro_callsite(cx, ex, "..").deref().to_string()));
            (
                "you don't need to add `&` to all patterns",
                "instead of prefixing all patterns with `&`, you can dereference the expression",
            )
        };

        suggs.extend(arms.iter().flat_map(|a| &a.pats).filter_map(|p| {
            if let PatKind::Ref(ref refp, _) = p.node {
                Some((p.span, snippet(cx, refp.span, "..").to_string()))
            } else {
                None
            }
        }));

        span_lint_and_then(cx, MATCH_REF_PATS, expr.span, title, |db| {
            if !in_macro_or_desugar(expr.span) {
                multispan_sugg(db, msg.to_owned(), suggs);
            }
        });
    }
}

fn check_match_as_ref(cx: &LateContext<'_, '_>, ex: &Expr, arms: &[Arm], expr: &Expr) {
    if arms.len() == 2
        && arms[0].pats.len() == 1
        && arms[0].guard.is_none()
        && arms[1].pats.len() == 1
        && arms[1].guard.is_none()
    {
        let arm_ref: Option<BindingAnnotation> = if is_none_arm(&arms[0]) {
            is_ref_some_arm(&arms[1])
        } else if is_none_arm(&arms[1]) {
            is_ref_some_arm(&arms[0])
        } else {
            None
        };
        if let Some(rb) = arm_ref {
            let suggestion = if rb == BindingAnnotation::Ref {
                "as_ref"
            } else {
                "as_mut"
            };
            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                MATCH_AS_REF,
                expr.span,
                &format!("use {}() instead", suggestion),
                "try this",
                format!(
                    "{}.{}()",
                    snippet_with_applicability(cx, ex.span, "_", &mut applicability),
                    suggestion
                ),
                applicability,
            )
        }
    }
}

/// Gets all arms that are unbounded `PatRange`s.
fn all_ranges<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, arms: &'tcx [Arm]) -> Vec<SpannedRange<Constant>> {
    arms.iter()
        .flat_map(|arm| {
            if let Arm {
                ref pats, guard: None, ..
            } = *arm
            {
                pats.iter()
            } else {
                [].iter()
            }
            .filter_map(|pat| {
                if let PatKind::Range(ref lhs, ref rhs, ref range_end) = pat.node {
                    let lhs = constant(cx, cx.tables, lhs)?.0;
                    let rhs = constant(cx, cx.tables, rhs)?.0;
                    let rhs = match *range_end {
                        RangeEnd::Included => Bound::Included(rhs),
                        RangeEnd::Excluded => Bound::Excluded(rhs),
                    };
                    return Some(SpannedRange {
                        span: pat.span,
                        node: (lhs, rhs),
                    });
                }

                if let PatKind::Lit(ref value) = pat.node {
                    let value = constant(cx, cx.tables, value)?.0;
                    return Some(SpannedRange {
                        span: pat.span,
                        node: (value.clone(), Bound::Included(value)),
                    });
                }

                None
            })
        })
        .collect()
}

#[derive(Debug, Eq, PartialEq)]
pub struct SpannedRange<T> {
    pub span: Span,
    pub node: (T, Bound<T>),
}

type TypedRanges = Vec<SpannedRange<u128>>;

/// Gets all `Int` ranges or all `Uint` ranges. Mixed types are an error anyway
/// and other types than
/// `Uint` and `Int` probably don't make sense.
fn type_ranges(ranges: &[SpannedRange<Constant>]) -> TypedRanges {
    ranges
        .iter()
        .filter_map(|range| match range.node {
            (Constant::Int(start), Bound::Included(Constant::Int(end))) => Some(SpannedRange {
                span: range.span,
                node: (start, Bound::Included(end)),
            }),
            (Constant::Int(start), Bound::Excluded(Constant::Int(end))) => Some(SpannedRange {
                span: range.span,
                node: (start, Bound::Excluded(end)),
            }),
            (Constant::Int(start), Bound::Unbounded) => Some(SpannedRange {
                span: range.span,
                node: (start, Bound::Unbounded),
            }),
            _ => None,
        })
        .collect()
}

fn is_unit_expr(expr: &Expr) -> bool {
    match expr.node {
        ExprKind::Tup(ref v) if v.is_empty() => true,
        ExprKind::Block(ref b, _) if b.stmts.is_empty() && b.expr.is_none() => true,
        _ => false,
    }
}

// Checks if arm has the form `None => None`
fn is_none_arm(arm: &Arm) -> bool {
    match arm.pats[0].node {
        PatKind::Path(ref path) if match_qpath(path, &paths::OPTION_NONE) => true,
        _ => false,
    }
}

// Checks if arm has the form `Some(ref v) => Some(v)` (checks for `ref` and `ref mut`)
fn is_ref_some_arm(arm: &Arm) -> Option<BindingAnnotation> {
    if_chain! {
        if let PatKind::TupleStruct(ref path, ref pats, _) = arm.pats[0].node;
        if pats.len() == 1 && match_qpath(path, &paths::OPTION_SOME);
        if let PatKind::Binding(rb, .., ident, _) = pats[0].node;
        if rb == BindingAnnotation::Ref || rb == BindingAnnotation::RefMut;
        if let ExprKind::Call(ref e, ref args) = remove_blocks(&arm.body).node;
        if let ExprKind::Path(ref some_path) = e.node;
        if match_qpath(some_path, &paths::OPTION_SOME) && args.len() == 1;
        if let ExprKind::Path(ref qpath) = args[0].node;
        if let &QPath::Resolved(_, ref path2) = qpath;
        if path2.segments.len() == 1 && ident.name == path2.segments[0].ident.name;
        then {
            return Some(rb)
        }
    }
    None
}

fn has_only_ref_pats(arms: &[Arm]) -> bool {
    let mapped = arms
        .iter()
        .flat_map(|a| &a.pats)
        .map(|p| {
            match p.node {
                PatKind::Ref(..) => Some(true), // &-patterns
                PatKind::Wild => Some(false),   // an "anything" wildcard is also fine
                _ => None,                      // any other pattern is not fine
            }
        })
        .collect::<Option<Vec<bool>>>();
    // look for Some(v) where there's at least one true element
    mapped.map_or(false, |v| v.iter().any(|el| *el))
}

pub fn overlapping<T>(ranges: &[SpannedRange<T>]) -> Option<(&SpannedRange<T>, &SpannedRange<T>)>
where
    T: Copy + Ord,
{
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    enum Kind<'a, T> {
        Start(T, &'a SpannedRange<T>),
        End(Bound<T>, &'a SpannedRange<T>),
    }

    impl<'a, T: Copy> Kind<'a, T> {
        fn range(&self) -> &'a SpannedRange<T> {
            match *self {
                Kind::Start(_, r) | Kind::End(_, r) => r,
            }
        }

        fn value(self) -> Bound<T> {
            match self {
                Kind::Start(t, _) => Bound::Included(t),
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
            match (self.value(), other.value()) {
                (Bound::Included(a), Bound::Included(b)) | (Bound::Excluded(a), Bound::Excluded(b)) => a.cmp(&b),
                // Range patterns cannot be unbounded (yet)
                (Bound::Unbounded, _) | (_, Bound::Unbounded) => unimplemented!(),
                (Bound::Included(a), Bound::Excluded(b)) => match a.cmp(&b) {
                    Ordering::Equal => Ordering::Greater,
                    other => other,
                },
                (Bound::Excluded(a), Bound::Included(b)) => match a.cmp(&b) {
                    Ordering::Equal => Ordering::Less,
                    other => other,
                },
            }
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
            },
            (&Kind::End(a, _), &Kind::Start(b, _)) if a != Bound::Included(b) => (),
            _ => return Some((a.range(), b.range())),
        }
    }

    None
}
