use crate::consts::{constant, miri_to_const, Constant};
use crate::utils::paths;
use crate::utils::sugg::Sugg;
use crate::utils::usage::is_unused;
use crate::utils::{
    expr_block, get_arg_name, get_parent_expr, in_macro, indent_of, is_allowed, is_expn_of, is_refutable,
    is_type_diagnostic_item, is_wild, match_qpath, match_type, match_var, multispan_sugg, remove_blocks, snippet,
    snippet_block, snippet_with_applicability, span_lint_and_help, span_lint_and_note, span_lint_and_sugg,
    span_lint_and_then, walk_ptrs_ty,
};
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::def::CtorKind;
use rustc_hir::{
    Arm, BindingAnnotation, Block, BorrowKind, Expr, ExprKind, Guard, Local, MatchSource, Mutability, Node, Pat,
    PatKind, QPath, RangeEnd,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{self, Ty};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::source_map::{Span, Spanned};
use std::cmp::Ordering;
use std::collections::Bound;

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
    ///
    /// // Bad
    /// match x {
    ///     Some(ref foo) => bar(foo),
    ///     _ => (),
    /// }
    ///
    /// // Good
    /// if let Some(ref foo) = x {
    ///     bar(foo);
    /// }
    /// ```
    pub SINGLE_MATCH,
    style,
    "a `match` statement with a single nontrivial arm (i.e., where the other arm is `_ => {}`) instead of `if let`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for matches with two arms where an `if let else` will
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
    /// # fn bar(foo: &usize) {}
    /// # let other_ref: usize = 1;
    /// # let x: Option<&usize> = Some(&1);
    /// match x {
    ///     Some(ref foo) => bar(foo),
    ///     _ => bar(&other_ref),
    /// }
    /// ```
    ///
    /// Using `if let` with `else`:
    ///
    /// ```rust
    /// # fn bar(foo: &usize) {}
    /// # let other_ref: usize = 1;
    /// # let x: Option<&usize> = Some(&1);
    /// if let Some(ref foo) = x {
    ///     bar(foo);
    /// } else {
    ///     bar(&other_ref);
    /// }
    /// ```
    pub SINGLE_MATCH_ELSE,
    pedantic,
    "a `match` statement with two arms where the second arm's pattern is a placeholder instead of a specific match pattern"
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
    /// // Bad
    /// match x {
    ///     &A(ref y) => foo(y),
    ///     &B => bar(),
    ///     _ => frob(&x),
    /// }
    ///
    /// // Good
    /// match *x {
    ///     A(ref y) => foo(y),
    ///     B => bar(),
    ///     _ => frob(x),
    /// }
    /// ```
    pub MATCH_REF_PATS,
    style,
    "a `match` or `if let` with all arms prefixed with `&` instead of deref-ing the match expression"
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
    pedantic,
    "a `match` on a boolean expression instead of an `if..else` block"
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
    "a `match` with overlapping arms"
}

declare_clippy_lint! {
    /// **What it does:** Checks for arm which matches all errors with `Err(_)`
    /// and take drastic actions like `panic!`.
    ///
    /// **Why is this bad?** It is generally a bad practice, similar to
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
    pedantic,
    "a `match` with `Err(_)` arm and take drastic actions"
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
    ///
    /// // Bad
    /// let r: Option<&()> = match x {
    ///     None => None,
    ///     Some(ref v) => Some(v),
    /// };
    ///
    /// // Good
    /// let r: Option<&()> = x.as_ref();
    /// ```
    pub MATCH_AS_REF,
    complexity,
    "a `match` on an Option value instead of using `as_ref()` or `as_mut`"
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
    /// # enum Foo { A(usize), B(usize) }
    /// # let x = Foo::B(1);
    ///
    /// // Bad
    /// match x {
    ///     Foo::A(_) => {},
    ///     _ => {},
    /// }
    ///
    /// // Good
    /// match x {
    ///     Foo::A(_) => {},
    ///     Foo::B(_) => {},
    /// }
    /// ```
    pub WILDCARD_ENUM_MATCH_ARM,
    restriction,
    "a wildcard enum match arm using `_`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for wildcard enum matches for a single variant.
    ///
    /// **Why is this bad?** New enum variants added by library updates can be missed.
    ///
    /// **Known problems:** Suggested replacements may not use correct path to enum
    /// if it's not present in the current scope.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// # enum Foo { A, B, C }
    /// # let x = Foo::B;
    /// // Bad
    /// match x {
    ///     Foo::A => {},
    ///     Foo::B => {},
    ///     _ => {},
    /// }
    ///
    /// // Good
    /// match x {
    ///     Foo::A => {},
    ///     Foo::B => {},
    ///     Foo::C => {},
    /// }
    /// ```
    pub MATCH_WILDCARD_FOR_SINGLE_VARIANTS,
    pedantic,
    "a wildcard enum match for a single variant"
}

declare_clippy_lint! {
    /// **What it does:** Checks for wildcard pattern used with others patterns in same match arm.
    ///
    /// **Why is this bad?** Wildcard pattern already covers any other pattern as it will match anyway.
    /// It makes the code less readable, especially to spot wildcard pattern use in match arm.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// // Bad
    /// match "foo" {
    ///     "a" => {},
    ///     "bar" | _ => {},
    /// }
    ///
    /// // Good
    /// match "foo" {
    ///     "a" => {},
    ///     _ => {},
    /// }
    /// ```
    pub WILDCARD_IN_OR_PATTERNS,
    complexity,
    "a wildcard pattern used with others patterns in same match arm"
}

declare_clippy_lint! {
    /// **What it does:** Checks for matches being used to destructure a single-variant enum
    /// or tuple struct where a `let` will suffice.
    ///
    /// **Why is this bad?** Just readability – `let` doesn't nest, whereas a `match` does.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// enum Wrapper {
    ///     Data(i32),
    /// }
    ///
    /// let wrapper = Wrapper::Data(42);
    ///
    /// let data = match wrapper {
    ///     Wrapper::Data(i) => i,
    /// };
    /// ```
    ///
    /// The correct use would be:
    /// ```rust
    /// enum Wrapper {
    ///     Data(i32),
    /// }
    ///
    /// let wrapper = Wrapper::Data(42);
    /// let Wrapper::Data(data) = wrapper;
    /// ```
    pub INFALLIBLE_DESTRUCTURING_MATCH,
    style,
    "a `match` statement with a single infallible arm instead of a `let`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for useless match that binds to only one value.
    ///
    /// **Why is this bad?** Readability and needless complexity.
    ///
    /// **Known problems:**  Suggested replacements may be incorrect when `match`
    /// is actually binding temporary value, bringing a 'dropped while borrowed' error.
    ///
    /// **Example:**
    /// ```rust
    /// # let a = 1;
    /// # let b = 2;
    ///
    /// // Bad
    /// match (a, b) {
    ///     (c, d) => {
    ///         // useless match
    ///     }
    /// }
    ///
    /// // Good
    /// let (c, d) = (a, b);
    /// ```
    pub MATCH_SINGLE_BINDING,
    complexity,
    "a match with a single binding instead of using `let` statement"
}

declare_clippy_lint! {
    /// **What it does:** Checks for unnecessary '..' pattern binding on struct when all fields are explicitly matched.
    ///
    /// **Why is this bad?** Correctness and readability. It's like having a wildcard pattern after
    /// matching all enum variants explicitly.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # struct A { a: i32 }
    /// let a = A { a: 5 };
    ///
    /// // Bad
    /// match a {
    ///     A { a: 5, .. } => {},
    ///     _ => {},
    /// }
    ///
    /// // Good
    /// match a {
    ///     A { a: 5 } => {},
    ///     _ => {},
    /// }
    /// ```
    pub REST_PAT_IN_FULLY_BOUND_STRUCTS,
    restriction,
    "a match on a struct that binds all fields but still uses the wildcard pattern"
}

declare_clippy_lint! {
    /// **What it does:** Lint for redundant pattern matching over `Result` or
    /// `Option`
    ///
    /// **Why is this bad?** It's more concise and clear to just use the proper
    /// utility function
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// if let Ok(_) = Ok::<i32, i32>(42) {}
    /// if let Err(_) = Err::<i32, i32>(42) {}
    /// if let None = None::<()> {}
    /// if let Some(_) = Some(42) {}
    /// match Ok::<i32, i32>(42) {
    ///     Ok(_) => true,
    ///     Err(_) => false,
    /// };
    /// ```
    ///
    /// The more idiomatic use would be:
    ///
    /// ```rust
    /// if Ok::<i32, i32>(42).is_ok() {}
    /// if Err::<i32, i32>(42).is_err() {}
    /// if None::<()>.is_none() {}
    /// if Some(42).is_some() {}
    /// Ok::<i32, i32>(42).is_ok();
    /// ```
    pub REDUNDANT_PATTERN_MATCHING,
    style,
    "use the proper utility function avoiding an `if let`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for `match`  or `if let` expressions producing a
    /// `bool` that could be written using `matches!`
    ///
    /// **Why is this bad?** Readability and needless complexity.
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    /// ```rust
    /// let x = Some(5);
    ///
    /// // Bad
    /// let a = match x {
    ///     Some(0) => true,
    ///     _ => false,
    /// };
    ///
    /// let a = if let Some(0) = x {
    ///     true
    /// } else {
    ///     false
    /// };
    ///
    /// // Good
    /// let a = matches!(x, Some(0));
    /// ```
    pub MATCH_LIKE_MATCHES_MACRO,
    style,
    "a match that could be written with the matches! macro"
}

#[derive(Default)]
pub struct Matches {
    infallible_destructuring_match_linted: bool,
}

impl_lint_pass!(Matches => [
    SINGLE_MATCH,
    MATCH_REF_PATS,
    MATCH_BOOL,
    SINGLE_MATCH_ELSE,
    MATCH_OVERLAPPING_ARM,
    MATCH_WILD_ERR_ARM,
    MATCH_AS_REF,
    WILDCARD_ENUM_MATCH_ARM,
    MATCH_WILDCARD_FOR_SINGLE_VARIANTS,
    WILDCARD_IN_OR_PATTERNS,
    MATCH_SINGLE_BINDING,
    INFALLIBLE_DESTRUCTURING_MATCH,
    REST_PAT_IN_FULLY_BOUND_STRUCTS,
    REDUNDANT_PATTERN_MATCHING,
    MATCH_LIKE_MATCHES_MACRO
]);

impl<'tcx> LateLintPass<'tcx> for Matches {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if in_external_macro(cx.sess(), expr.span) {
            return;
        }

        redundant_pattern_match::check(cx, expr);
        check_match_like_matches(cx, expr);

        if let ExprKind::Match(ref ex, ref arms, MatchSource::Normal) = expr.kind {
            check_single_match(cx, ex, arms, expr);
            check_match_bool(cx, ex, arms, expr);
            check_overlapping_arms(cx, ex, arms);
            check_wild_err_arm(cx, ex, arms);
            check_wild_enum_match(cx, ex, arms);
            check_match_as_ref(cx, ex, arms, expr);
            check_wild_in_or_pats(cx, arms);

            if self.infallible_destructuring_match_linted {
                self.infallible_destructuring_match_linted = false;
            } else {
                check_match_single_binding(cx, ex, arms, expr);
            }
        }
        if let ExprKind::Match(ref ex, ref arms, _) = expr.kind {
            check_match_ref_pats(cx, ex, arms, expr);
        }
    }

    fn check_local(&mut self, cx: &LateContext<'tcx>, local: &'tcx Local<'_>) {
        if_chain! {
            if !in_external_macro(cx.sess(), local.span);
            if !in_macro(local.span);
            if let Some(ref expr) = local.init;
            if let ExprKind::Match(ref target, ref arms, MatchSource::Normal) = expr.kind;
            if arms.len() == 1 && arms[0].guard.is_none();
            if let PatKind::TupleStruct(
                QPath::Resolved(None, ref variant_name), ref args, _) = arms[0].pat.kind;
            if args.len() == 1;
            if let Some(arg) = get_arg_name(&args[0]);
            let body = remove_blocks(&arms[0].body);
            if match_var(body, arg);

            then {
                let mut applicability = Applicability::MachineApplicable;
                self.infallible_destructuring_match_linted = true;
                span_lint_and_sugg(
                    cx,
                    INFALLIBLE_DESTRUCTURING_MATCH,
                    local.span,
                    "you seem to be trying to use `match` to destructure a single infallible pattern. \
                    Consider using `let`",
                    "try this",
                    format!(
                        "let {}({}) = {};",
                        snippet_with_applicability(cx, variant_name.span, "..", &mut applicability),
                        snippet_with_applicability(cx, local.pat.span, "..", &mut applicability),
                        snippet_with_applicability(cx, target.span, "..", &mut applicability),
                    ),
                    applicability,
                );
            }
        }
    }

    fn check_pat(&mut self, cx: &LateContext<'tcx>, pat: &'tcx Pat<'_>) {
        if_chain! {
            if !in_external_macro(cx.sess(), pat.span);
            if !in_macro(pat.span);
            if let PatKind::Struct(ref qpath, fields, true) = pat.kind;
            if let QPath::Resolved(_, ref path) = qpath;
            if let Some(def_id) = path.res.opt_def_id();
            let ty = cx.tcx.type_of(def_id);
            if let ty::Adt(def, _) = ty.kind();
            if def.is_struct() || def.is_union();
            if fields.len() == def.non_enum_variant().fields.len();

            then {
                span_lint_and_help(
                    cx,
                    REST_PAT_IN_FULLY_BOUND_STRUCTS,
                    pat.span,
                    "unnecessary use of `..` pattern in struct binding. All fields were already bound",
                    None,
                    "consider removing `..` from this binding",
                );
            }
        }
    }
}

#[rustfmt::skip]
fn check_single_match(cx: &LateContext<'_>, ex: &Expr<'_>, arms: &[Arm<'_>], expr: &Expr<'_>) {
    if arms.len() == 2 && arms[0].guard.is_none() && arms[1].guard.is_none() {
        if in_macro(expr.span) {
            // Don't lint match expressions present in
            // macro_rules! block
            return;
        }
        if let PatKind::Or(..) = arms[0].pat.kind {
            // don't lint for or patterns for now, this makes
            // the lint noisy in unnecessary situations
            return;
        }
        let els = arms[1].body;
        let els = if is_unit_expr(remove_blocks(els)) {
            None
        } else if let ExprKind::Block(Block { stmts, expr: block_expr, .. }, _) = els.kind {
            if stmts.len() == 1 && block_expr.is_none() || stmts.is_empty() && block_expr.is_some() {
                // single statement/expr "else" block, don't lint
                return;
            } else {
                // block with 2+ statements or 1 expr and 1+ statement
                Some(els)
            }
        } else {
            // not a block, don't lint
            return; 
        };

        let ty = cx.typeck_results().expr_ty(ex);
        if *ty.kind() != ty::Bool || is_allowed(cx, MATCH_BOOL, ex.hir_id) {
            check_single_match_single_pattern(cx, ex, arms, expr, els);
            check_single_match_opt_like(cx, ex, arms, expr, ty, els);
        }
    }
}

fn check_single_match_single_pattern(
    cx: &LateContext<'_>,
    ex: &Expr<'_>,
    arms: &[Arm<'_>],
    expr: &Expr<'_>,
    els: Option<&Expr<'_>>,
) {
    if is_wild(&arms[1].pat) {
        report_single_match_single_pattern(cx, ex, arms, expr, els);
    }
}

fn report_single_match_single_pattern(
    cx: &LateContext<'_>,
    ex: &Expr<'_>,
    arms: &[Arm<'_>],
    expr: &Expr<'_>,
    els: Option<&Expr<'_>>,
) {
    let lint = if els.is_some() { SINGLE_MATCH_ELSE } else { SINGLE_MATCH };
    let els_str = els.map_or(String::new(), |els| {
        format!(" else {}", expr_block(cx, els, None, "..", Some(expr.span)))
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
            snippet(cx, arms[0].pat.span, ".."),
            snippet(cx, ex.span, ".."),
            expr_block(cx, &arms[0].body, None, "..", Some(expr.span)),
            els_str,
        ),
        Applicability::HasPlaceholders,
    );
}

fn check_single_match_opt_like(
    cx: &LateContext<'_>,
    ex: &Expr<'_>,
    arms: &[Arm<'_>],
    expr: &Expr<'_>,
    ty: Ty<'_>,
    els: Option<&Expr<'_>>,
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

    let path = match arms[1].pat.kind {
        PatKind::TupleStruct(ref path, ref inner, _) => {
            // Contains any non wildcard patterns (e.g., `Err(err)`)?
            if !inner.iter().all(is_wild) {
                return;
            }
            rustc_hir_pretty::to_string(rustc_hir_pretty::NO_ANN, |s| s.print_qpath(path, false))
        },
        PatKind::Binding(BindingAnnotation::Unannotated, .., ident, None) => ident.to_string(),
        PatKind::Path(ref path) => {
            rustc_hir_pretty::to_string(rustc_hir_pretty::NO_ANN, |s| s.print_qpath(path, false))
        },
        _ => return,
    };

    for &(ty_path, pat_path) in candidates {
        if path == *pat_path && match_type(cx, ty, ty_path) {
            report_single_match_single_pattern(cx, ex, arms, expr, els);
        }
    }
}

fn check_match_bool(cx: &LateContext<'_>, ex: &Expr<'_>, arms: &[Arm<'_>], expr: &Expr<'_>) {
    // Type of expression is `bool`.
    if *cx.typeck_results().expr_ty(ex).kind() == ty::Bool {
        span_lint_and_then(
            cx,
            MATCH_BOOL,
            expr.span,
            "you seem to be trying to match on a boolean expression",
            move |diag| {
                if arms.len() == 2 {
                    // no guards
                    let exprs = if let PatKind::Lit(ref arm_bool) = arms[0].pat.kind {
                        if let ExprKind::Lit(ref lit) = arm_bool.kind {
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
                                expr_block(cx, true_expr, None, "..", Some(expr.span)),
                                expr_block(cx, false_expr, None, "..", Some(expr.span))
                            )),
                            (false, true) => Some(format!(
                                "if {} {}",
                                snippet(cx, ex.span, "b"),
                                expr_block(cx, true_expr, None, "..", Some(expr.span))
                            )),
                            (true, false) => {
                                let test = Sugg::hir(cx, ex, "..");
                                Some(format!(
                                    "if {} {}",
                                    !test,
                                    expr_block(cx, false_expr, None, "..", Some(expr.span))
                                ))
                            },
                            (true, true) => None,
                        };

                        if let Some(sugg) = sugg {
                            diag.span_suggestion(
                                expr.span,
                                "consider using an `if`/`else` expression",
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

fn check_overlapping_arms<'tcx>(cx: &LateContext<'tcx>, ex: &'tcx Expr<'_>, arms: &'tcx [Arm<'_>]) {
    if arms.len() >= 2 && cx.typeck_results().expr_ty(ex).is_integral() {
        let ranges = all_ranges(cx, arms, cx.typeck_results().expr_ty(ex));
        let type_ranges = type_ranges(&ranges);
        if !type_ranges.is_empty() {
            if let Some((start, end)) = overlapping(&type_ranges) {
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

fn check_wild_err_arm(cx: &LateContext<'_>, ex: &Expr<'_>, arms: &[Arm<'_>]) {
    let ex_ty = walk_ptrs_ty(cx.typeck_results().expr_ty(ex));
    if is_type_diagnostic_item(cx, ex_ty, sym!(result_type)) {
        for arm in arms {
            if let PatKind::TupleStruct(ref path, ref inner, _) = arm.pat.kind {
                let path_str = rustc_hir_pretty::to_string(rustc_hir_pretty::NO_ANN, |s| s.print_qpath(path, false));
                if path_str == "Err" {
                    let mut matching_wild = inner.iter().any(is_wild);
                    let mut ident_bind_name = String::from("_");
                    if !matching_wild {
                        // Looking for unused bindings (i.e.: `_e`)
                        inner.iter().for_each(|pat| {
                            if let PatKind::Binding(.., ident, None) = &pat.kind {
                                if ident.as_str().starts_with('_') && is_unused(ident, arm.body) {
                                    ident_bind_name = (&ident.name.as_str()).to_string();
                                    matching_wild = true;
                                }
                            }
                        });
                    }
                    if_chain! {
                        if matching_wild;
                        if let ExprKind::Block(ref block, _) = arm.body.kind;
                        if is_panic_block(block);
                        then {
                            // `Err(_)` or `Err(_e)` arm with `panic!` found
                            span_lint_and_note(cx,
                                MATCH_WILD_ERR_ARM,
                                arm.pat.span,
                                &format!("`Err({})` matches all errors", &ident_bind_name),
                                None,
                                "match each error separately or use the error output, or use `.except(msg)` if the error case is unreachable",
                            );
                        }
                    }
                }
            }
        }
    }
}

fn check_wild_enum_match(cx: &LateContext<'_>, ex: &Expr<'_>, arms: &[Arm<'_>]) {
    let ty = cx.typeck_results().expr_ty(ex);
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
        if let PatKind::Wild = arm.pat.kind {
            wildcard_span = Some(arm.pat.span);
        } else if let PatKind::Binding(_, _, ident, None) = arm.pat.kind {
            wildcard_span = Some(arm.pat.span);
            wildcard_ident = Some(ident);
        }
    }

    if let Some(wildcard_span) = wildcard_span {
        // Accumulate the variants which should be put in place of the wildcard because they're not
        // already covered.

        let mut missing_variants = vec![];
        if let ty::Adt(def, _) = ty.kind() {
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
            if let PatKind::Path(ref path) = arm.pat.kind {
                if let QPath::Resolved(_, p) = path {
                    missing_variants.retain(|e| e.ctor_def_id != Some(p.res.def_id()));
                }
            } else if let PatKind::TupleStruct(ref path, ref patterns, ..) = arm.pat.kind {
                if let QPath::Resolved(_, p) = path {
                    // Some simple checks for exhaustive patterns.
                    // There is a room for improvements to detect more cases,
                    // but it can be more expensive to do so.
                    let is_pattern_exhaustive =
                        |pat: &&Pat<'_>| matches!(pat.kind, PatKind::Wild | PatKind::Binding(.., None));
                    if patterns.iter().all(is_pattern_exhaustive) {
                        missing_variants.retain(|e| e.ctor_def_id != Some(p.res.def_id()));
                    }
                }
            }
        }

        let mut suggestion: Vec<String> = missing_variants
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

        let mut message = "wildcard match will miss any future added variants";

        if let ty::Adt(def, _) = ty.kind() {
            if def.is_variant_list_non_exhaustive() {
                message = "match on non-exhaustive enum doesn't explicitly match all known variants";
                suggestion.push(String::from("_"));
            }
        }

        if suggestion.len() == 1 {
            // No need to check for non-exhaustive enum as in that case len would be greater than 1
            span_lint_and_sugg(
                cx,
                MATCH_WILDCARD_FOR_SINGLE_VARIANTS,
                wildcard_span,
                message,
                "try this",
                suggestion[0].clone(),
                Applicability::MaybeIncorrect,
            )
        };

        span_lint_and_sugg(
            cx,
            WILDCARD_ENUM_MATCH_ARM,
            wildcard_span,
            message,
            "try this",
            suggestion.join(" | "),
            Applicability::MaybeIncorrect,
        )
    }
}

// If the block contains only a `panic!` macro (as expression or statement)
fn is_panic_block(block: &Block<'_>) -> bool {
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

fn check_match_ref_pats(cx: &LateContext<'_>, ex: &Expr<'_>, arms: &[Arm<'_>], expr: &Expr<'_>) {
    if has_only_ref_pats(arms) {
        let mut suggs = Vec::with_capacity(arms.len() + 1);
        let (title, msg) = if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, ref inner) = ex.kind {
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

        suggs.extend(arms.iter().filter_map(|a| {
            if let PatKind::Ref(ref refp, _) = a.pat.kind {
                Some((a.pat.span, snippet(cx, refp.span, "..").to_string()))
            } else {
                None
            }
        }));

        span_lint_and_then(cx, MATCH_REF_PATS, expr.span, title, |diag| {
            if !expr.span.from_expansion() {
                multispan_sugg(diag, msg, suggs);
            }
        });
    }
}

fn check_match_as_ref(cx: &LateContext<'_>, ex: &Expr<'_>, arms: &[Arm<'_>], expr: &Expr<'_>) {
    if arms.len() == 2 && arms[0].guard.is_none() && arms[1].guard.is_none() {
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

            let output_ty = cx.typeck_results().expr_ty(expr);
            let input_ty = cx.typeck_results().expr_ty(ex);

            let cast = if_chain! {
                if let ty::Adt(_, substs) = input_ty.kind();
                let input_ty = substs.type_at(0);
                if let ty::Adt(_, substs) = output_ty.kind();
                let output_ty = substs.type_at(0);
                if let ty::Ref(_, output_ty, _) = *output_ty.kind();
                if input_ty != output_ty;
                then {
                    ".map(|x| x as _)"
                } else {
                    ""
                }
            };

            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                MATCH_AS_REF,
                expr.span,
                &format!("use `{}()` instead", suggestion),
                "try this",
                format!(
                    "{}.{}(){}",
                    snippet_with_applicability(cx, ex.span, "_", &mut applicability),
                    suggestion,
                    cast,
                ),
                applicability,
            )
        }
    }
}

fn check_wild_in_or_pats(cx: &LateContext<'_>, arms: &[Arm<'_>]) {
    for arm in arms {
        if let PatKind::Or(ref fields) = arm.pat.kind {
            // look for multiple fields in this arm that contains at least one Wild pattern
            if fields.len() > 1 && fields.iter().any(is_wild) {
                span_lint_and_help(
                    cx,
                    WILDCARD_IN_OR_PATTERNS,
                    arm.pat.span,
                    "wildcard pattern covers any other pattern as it will match anyway.",
                    None,
                    "Consider handling `_` separately.",
                );
            }
        }
    }
}

/// Lint a `match` or `if let .. { .. } else { .. }` expr that could be replaced by `matches!`
fn check_match_like_matches<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
    if let ExprKind::Match(ex, arms, ref match_source) = &expr.kind {
        match match_source {
            MatchSource::Normal => find_matches_sugg(cx, ex, arms, expr, false),
            MatchSource::IfLetDesugar { .. } => find_matches_sugg(cx, ex, arms, expr, true),
            _ => return,
        }
    }
}

/// Lint a `match` or desugared `if let` for replacement by `matches!`
fn find_matches_sugg(cx: &LateContext<'_>, ex: &Expr<'_>, arms: &[Arm<'_>], expr: &Expr<'_>, desugared: bool) {
    if_chain! {
        if arms.len() == 2;
        if cx.typeck_results().expr_ty(expr).is_bool();
        if is_wild(&arms[1].pat);
        if let Some(first) = find_bool_lit(&arms[0].body.kind, desugared);
        if let Some(second) = find_bool_lit(&arms[1].body.kind, desugared);
        if first != second;
        then {
            let mut applicability = Applicability::MachineApplicable;

            let pat_and_guard = if let Some(Guard::If(g)) = arms[0].guard {
                format!("{} if {}", snippet_with_applicability(cx, arms[0].pat.span, "..", &mut applicability), snippet_with_applicability(cx, g.span, "..", &mut applicability))
            } else {
                format!("{}", snippet_with_applicability(cx, arms[0].pat.span, "..", &mut applicability))
            };
            span_lint_and_sugg(
                cx,
                MATCH_LIKE_MATCHES_MACRO,
                expr.span,
                &format!("{} expression looks like `matches!` macro", if desugared { "if let .. else" } else { "match" }),
                "try this",
                format!(
                    "{}matches!({}, {})",
                    if first { "" } else { "!" },
                    snippet_with_applicability(cx, ex.span, "..", &mut applicability),
                    pat_and_guard,
                ),
                applicability,
            )
        }
    }
}

/// Extract a `bool` or `{ bool }`
fn find_bool_lit(ex: &ExprKind<'_>, desugared: bool) -> Option<bool> {
    match ex {
        ExprKind::Lit(Spanned {
            node: LitKind::Bool(b), ..
        }) => Some(*b),
        ExprKind::Block(
            rustc_hir::Block {
                stmts: &[],
                expr: Some(exp),
                ..
            },
            _,
        ) if desugared => {
            if let ExprKind::Lit(Spanned {
                node: LitKind::Bool(b), ..
            }) = exp.kind
            {
                Some(b)
            } else {
                None
            }
        },
        _ => None,
    }
}

fn check_match_single_binding<'a>(cx: &LateContext<'a>, ex: &Expr<'a>, arms: &[Arm<'_>], expr: &Expr<'_>) {
    if in_macro(expr.span) || arms.len() != 1 || is_refutable(cx, arms[0].pat) {
        return;
    }
    let matched_vars = ex.span;
    let bind_names = arms[0].pat.span;
    let match_body = remove_blocks(&arms[0].body);
    let mut snippet_body = if match_body.span.from_expansion() {
        Sugg::hir_with_macro_callsite(cx, match_body, "..").to_string()
    } else {
        snippet_block(cx, match_body.span, "..", Some(expr.span)).to_string()
    };

    // Do we need to add ';' to suggestion ?
    match match_body.kind {
        ExprKind::Block(block, _) => {
            // macro + expr_ty(body) == ()
            if block.span.from_expansion() && cx.typeck_results().expr_ty(&match_body).is_unit() {
                snippet_body.push(';');
            }
        },
        _ => {
            // expr_ty(body) == ()
            if cx.typeck_results().expr_ty(&match_body).is_unit() {
                snippet_body.push(';');
            }
        },
    }

    let mut applicability = Applicability::MaybeIncorrect;
    match arms[0].pat.kind {
        PatKind::Binding(..) | PatKind::Tuple(_, _) | PatKind::Struct(..) => {
            // If this match is in a local (`let`) stmt
            let (target_span, sugg) = if let Some(parent_let_node) = opt_parent_let(cx, ex) {
                (
                    parent_let_node.span,
                    format!(
                        "let {} = {};\n{}let {} = {};",
                        snippet_with_applicability(cx, bind_names, "..", &mut applicability),
                        snippet_with_applicability(cx, matched_vars, "..", &mut applicability),
                        " ".repeat(indent_of(cx, expr.span).unwrap_or(0)),
                        snippet_with_applicability(cx, parent_let_node.pat.span, "..", &mut applicability),
                        snippet_body
                    ),
                )
            } else {
                // If we are in closure, we need curly braces around suggestion
                let mut indent = " ".repeat(indent_of(cx, ex.span).unwrap_or(0));
                let (mut cbrace_start, mut cbrace_end) = ("".to_string(), "".to_string());
                if let Some(parent_expr) = get_parent_expr(cx, expr) {
                    if let ExprKind::Closure(..) = parent_expr.kind {
                        cbrace_end = format!("\n{}}}", indent);
                        // Fix body indent due to the closure
                        indent = " ".repeat(indent_of(cx, bind_names).unwrap_or(0));
                        cbrace_start = format!("{{\n{}", indent);
                    }
                };
                (
                    expr.span,
                    format!(
                        "{}let {} = {};\n{}{}{}",
                        cbrace_start,
                        snippet_with_applicability(cx, bind_names, "..", &mut applicability),
                        snippet_with_applicability(cx, matched_vars, "..", &mut applicability),
                        indent,
                        snippet_body,
                        cbrace_end
                    ),
                )
            };
            span_lint_and_sugg(
                cx,
                MATCH_SINGLE_BINDING,
                target_span,
                "this match could be written as a `let` statement",
                "consider using `let` statement",
                sugg,
                applicability,
            );
        },
        PatKind::Wild => {
            span_lint_and_sugg(
                cx,
                MATCH_SINGLE_BINDING,
                expr.span,
                "this match could be replaced by its body itself",
                "consider using the match body instead",
                snippet_body,
                Applicability::MachineApplicable,
            );
        },
        _ => (),
    }
}

/// Returns true if the `ex` match expression is in a local (`let`) statement
fn opt_parent_let<'a>(cx: &LateContext<'a>, ex: &Expr<'a>) -> Option<&'a Local<'a>> {
    if_chain! {
        let map = &cx.tcx.hir();
        if let Some(Node::Expr(parent_arm_expr)) = map.find(map.get_parent_node(ex.hir_id));
        if let Some(Node::Local(parent_let_expr)) = map.find(map.get_parent_node(parent_arm_expr.hir_id));
        then {
            return Some(parent_let_expr);
        }
    }
    None
}

/// Gets all arms that are unbounded `PatRange`s.
fn all_ranges<'tcx>(cx: &LateContext<'tcx>, arms: &'tcx [Arm<'_>], ty: Ty<'tcx>) -> Vec<SpannedRange<Constant>> {
    arms.iter()
        .flat_map(|arm| {
            if let Arm {
                ref pat, guard: None, ..
            } = *arm
            {
                if let PatKind::Range(ref lhs, ref rhs, range_end) = pat.kind {
                    let lhs = match lhs {
                        Some(lhs) => constant(cx, cx.typeck_results(), lhs)?.0,
                        None => miri_to_const(ty.numeric_min_val(cx.tcx)?)?,
                    };
                    let rhs = match rhs {
                        Some(rhs) => constant(cx, cx.typeck_results(), rhs)?.0,
                        None => miri_to_const(ty.numeric_max_val(cx.tcx)?)?,
                    };
                    let rhs = match range_end {
                        RangeEnd::Included => Bound::Included(rhs),
                        RangeEnd::Excluded => Bound::Excluded(rhs),
                    };
                    return Some(SpannedRange {
                        span: pat.span,
                        node: (lhs, rhs),
                    });
                }

                if let PatKind::Lit(ref value) = pat.kind {
                    let value = constant(cx, cx.typeck_results(), value)?.0;
                    return Some(SpannedRange {
                        span: pat.span,
                        node: (value.clone(), Bound::Included(value)),
                    });
                }
            }
            None
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

fn is_unit_expr(expr: &Expr<'_>) -> bool {
    match expr.kind {
        ExprKind::Tup(ref v) if v.is_empty() => true,
        ExprKind::Block(ref b, _) if b.stmts.is_empty() && b.expr.is_none() => true,
        _ => false,
    }
}

// Checks if arm has the form `None => None`
fn is_none_arm(arm: &Arm<'_>) -> bool {
    matches!(arm.pat.kind, PatKind::Path(ref path) if match_qpath(path, &paths::OPTION_NONE))
}

// Checks if arm has the form `Some(ref v) => Some(v)` (checks for `ref` and `ref mut`)
fn is_ref_some_arm(arm: &Arm<'_>) -> Option<BindingAnnotation> {
    if_chain! {
        if let PatKind::TupleStruct(ref path, ref pats, _) = arm.pat.kind;
        if pats.len() == 1 && match_qpath(path, &paths::OPTION_SOME);
        if let PatKind::Binding(rb, .., ident, _) = pats[0].kind;
        if rb == BindingAnnotation::Ref || rb == BindingAnnotation::RefMut;
        if let ExprKind::Call(ref e, ref args) = remove_blocks(&arm.body).kind;
        if let ExprKind::Path(ref some_path) = e.kind;
        if match_qpath(some_path, &paths::OPTION_SOME) && args.len() == 1;
        if let ExprKind::Path(ref qpath) = args[0].kind;
        if let &QPath::Resolved(_, ref path2) = qpath;
        if path2.segments.len() == 1 && ident.name == path2.segments[0].ident.name;
        then {
            return Some(rb)
        }
    }
    None
}

fn has_only_ref_pats(arms: &[Arm<'_>]) -> bool {
    let mapped = arms
        .iter()
        .map(|a| {
            match a.pat.kind {
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

mod redundant_pattern_match {
    use super::REDUNDANT_PATTERN_MATCHING;
    use crate::utils::{in_constant, match_qpath, match_trait_method, paths, snippet, span_lint_and_then};
    use if_chain::if_chain;
    use rustc_ast::ast::LitKind;
    use rustc_errors::Applicability;
    use rustc_hir::{Arm, Expr, ExprKind, HirId, MatchSource, PatKind, QPath};
    use rustc_lint::LateContext;
    use rustc_middle::ty;
    use rustc_mir::const_eval::is_const_fn;
    use rustc_span::source_map::Symbol;

    pub fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Match(op, arms, ref match_source) = &expr.kind {
            match match_source {
                MatchSource::Normal => find_sugg_for_match(cx, expr, op, arms),
                MatchSource::IfLetDesugar { .. } => find_sugg_for_if_let(cx, expr, op, arms, "if"),
                MatchSource::WhileLetDesugar => find_sugg_for_if_let(cx, expr, op, arms, "while"),
                _ => {},
            }
        }
    }

    fn find_sugg_for_if_let<'tcx>(
        cx: &LateContext<'tcx>,
        expr: &'tcx Expr<'_>,
        op: &Expr<'_>,
        arms: &[Arm<'_>],
        keyword: &'static str,
    ) {
        fn find_suggestion(cx: &LateContext<'_>, hir_id: HirId, path: &QPath<'_>) -> Option<&'static str> {
            if match_qpath(path, &paths::RESULT_OK) && can_suggest(cx, hir_id, sym!(result_type), "is_ok") {
                return Some("is_ok()");
            }
            if match_qpath(path, &paths::RESULT_ERR) && can_suggest(cx, hir_id, sym!(result_type), "is_err") {
                return Some("is_err()");
            }
            if match_qpath(path, &paths::OPTION_SOME) && can_suggest(cx, hir_id, sym!(option_type), "is_some") {
                return Some("is_some()");
            }
            if match_qpath(path, &paths::OPTION_NONE) && can_suggest(cx, hir_id, sym!(option_type), "is_none") {
                return Some("is_none()");
            }
            None
        }

        let hir_id = expr.hir_id;
        let good_method = match arms[0].pat.kind {
            PatKind::TupleStruct(ref path, ref patterns, _) if patterns.len() == 1 => {
                if let PatKind::Wild = patterns[0].kind {
                    find_suggestion(cx, hir_id, path)
                } else {
                    None
                }
            },
            PatKind::Path(ref path) => find_suggestion(cx, hir_id, path),
            _ => None,
        };
        let good_method = match good_method {
            Some(method) => method,
            None => return,
        };

        // check that `while_let_on_iterator` lint does not trigger
        if_chain! {
            if keyword == "while";
            if let ExprKind::MethodCall(method_path, _, _, _) = op.kind;
            if method_path.ident.name == sym!(next);
            if match_trait_method(cx, op, &paths::ITERATOR);
            then {
                return;
            }
        }

        let result_expr = match &op.kind {
            ExprKind::AddrOf(_, _, borrowed) => borrowed,
            _ => op,
        };
        span_lint_and_then(
            cx,
            REDUNDANT_PATTERN_MATCHING,
            arms[0].pat.span,
            &format!("redundant pattern matching, consider using `{}`", good_method),
            |diag| {
                // while let ... = ... { ... }
                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                let expr_span = expr.span;

                // while let ... = ... { ... }
                //                 ^^^
                let op_span = result_expr.span.source_callsite();

                // while let ... = ... { ... }
                // ^^^^^^^^^^^^^^^^^^^
                let span = expr_span.until(op_span.shrink_to_hi());
                diag.span_suggestion(
                    span,
                    "try this",
                    format!("{} {}.{}", keyword, snippet(cx, op_span, "_"), good_method),
                    Applicability::MachineApplicable, // snippet
                );
            },
        );
    }

    fn find_sugg_for_match<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, op: &Expr<'_>, arms: &[Arm<'_>]) {
        if arms.len() == 2 {
            let node_pair = (&arms[0].pat.kind, &arms[1].pat.kind);

            let hir_id = expr.hir_id;
            let found_good_method = match node_pair {
                (
                    PatKind::TupleStruct(ref path_left, ref patterns_left, _),
                    PatKind::TupleStruct(ref path_right, ref patterns_right, _),
                ) if patterns_left.len() == 1 && patterns_right.len() == 1 => {
                    if let (PatKind::Wild, PatKind::Wild) = (&patterns_left[0].kind, &patterns_right[0].kind) {
                        find_good_method_for_match(
                            arms,
                            path_left,
                            path_right,
                            &paths::RESULT_OK,
                            &paths::RESULT_ERR,
                            "is_ok()",
                            "is_err()",
                            || can_suggest(cx, hir_id, sym!(result_type), "is_ok"),
                            || can_suggest(cx, hir_id, sym!(result_type), "is_err"),
                        )
                    } else {
                        None
                    }
                },
                (PatKind::TupleStruct(ref path_left, ref patterns, _), PatKind::Path(ref path_right))
                | (PatKind::Path(ref path_left), PatKind::TupleStruct(ref path_right, ref patterns, _))
                    if patterns.len() == 1 =>
                {
                    if let PatKind::Wild = patterns[0].kind {
                        find_good_method_for_match(
                            arms,
                            path_left,
                            path_right,
                            &paths::OPTION_SOME,
                            &paths::OPTION_NONE,
                            "is_some()",
                            "is_none()",
                            || can_suggest(cx, hir_id, sym!(option_type), "is_some"),
                            || can_suggest(cx, hir_id, sym!(option_type), "is_none"),
                        )
                    } else {
                        None
                    }
                },
                _ => None,
            };

            if let Some(good_method) = found_good_method {
                let span = expr.span.to(op.span);
                let result_expr = match &op.kind {
                    ExprKind::AddrOf(_, _, borrowed) => borrowed,
                    _ => op,
                };
                span_lint_and_then(
                    cx,
                    REDUNDANT_PATTERN_MATCHING,
                    expr.span,
                    &format!("redundant pattern matching, consider using `{}`", good_method),
                    |diag| {
                        diag.span_suggestion(
                            span,
                            "try this",
                            format!("{}.{}", snippet(cx, result_expr.span, "_"), good_method),
                            Applicability::MaybeIncorrect, // snippet
                        );
                    },
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn find_good_method_for_match<'a>(
        arms: &[Arm<'_>],
        path_left: &QPath<'_>,
        path_right: &QPath<'_>,
        expected_left: &[&str],
        expected_right: &[&str],
        should_be_left: &'a str,
        should_be_right: &'a str,
        can_suggest_left: impl Fn() -> bool,
        can_suggest_right: impl Fn() -> bool,
    ) -> Option<&'a str> {
        let body_node_pair = if match_qpath(path_left, expected_left) && match_qpath(path_right, expected_right) {
            (&(*arms[0].body).kind, &(*arms[1].body).kind)
        } else if match_qpath(path_right, expected_left) && match_qpath(path_left, expected_right) {
            (&(*arms[1].body).kind, &(*arms[0].body).kind)
        } else {
            return None;
        };

        match body_node_pair {
            (ExprKind::Lit(ref lit_left), ExprKind::Lit(ref lit_right)) => match (&lit_left.node, &lit_right.node) {
                (LitKind::Bool(true), LitKind::Bool(false)) if can_suggest_left() => Some(should_be_left),
                (LitKind::Bool(false), LitKind::Bool(true)) if can_suggest_right() => Some(should_be_right),
                _ => None,
            },
            _ => None,
        }
    }

    fn can_suggest(cx: &LateContext<'_>, hir_id: HirId, diag_item: Symbol, name: &str) -> bool {
        if !in_constant(cx, hir_id) {
            return true;
        }

        // Avoid suggesting calls to non-`const fn`s in const contexts, see #5697.
        cx.tcx
            .get_diagnostic_item(diag_item)
            .and_then(|def_id| {
                cx.tcx.inherent_impls(def_id).iter().find_map(|imp| {
                    cx.tcx
                        .associated_items(*imp)
                        .in_definition_order()
                        .find_map(|item| match item.kind {
                            ty::AssocKind::Fn if item.ident.name.as_str() == name => Some(item.def_id),
                            _ => None,
                        })
                })
            })
            .map_or(false, |def_id| is_const_fn(cx.tcx, def_id))
    }
}

#[test]
fn test_overlapping() {
    use rustc_span::source_map::DUMMY_SP;

    let sp = |s, e| SpannedRange {
        span: DUMMY_SP,
        node: (s, e),
    };

    assert_eq!(None, overlapping::<u8>(&[]));
    assert_eq!(None, overlapping(&[sp(1, Bound::Included(4))]));
    assert_eq!(
        None,
        overlapping(&[sp(1, Bound::Included(4)), sp(5, Bound::Included(6))])
    );
    assert_eq!(
        None,
        overlapping(&[
            sp(1, Bound::Included(4)),
            sp(5, Bound::Included(6)),
            sp(10, Bound::Included(11))
        ],)
    );
    assert_eq!(
        Some((&sp(1, Bound::Included(4)), &sp(3, Bound::Included(6)))),
        overlapping(&[sp(1, Bound::Included(4)), sp(3, Bound::Included(6))])
    );
    assert_eq!(
        Some((&sp(5, Bound::Included(6)), &sp(6, Bound::Included(11)))),
        overlapping(&[
            sp(1, Bound::Included(4)),
            sp(5, Bound::Included(6)),
            sp(6, Bound::Included(11))
        ],)
    );
}
