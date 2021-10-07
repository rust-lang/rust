use clippy_utils::consts::{constant, miri_to_const, Constant};
use clippy_utils::diagnostics::{
    multispan_sugg, span_lint_and_help, span_lint_and_note, span_lint_and_sugg, span_lint_and_then,
};
use clippy_utils::higher;
use clippy_utils::source::{expr_block, indent_of, snippet, snippet_block, snippet_opt, snippet_with_applicability};
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::{implements_trait, is_type_diagnostic_item, match_type, peel_mid_ty_refs};
use clippy_utils::visitors::is_local_used;
use clippy_utils::{
    get_parent_expr, in_macro, is_expn_of, is_lang_ctor, is_lint_allowed, is_refutable, is_unit_expr, is_wild,
    meets_msrv, msrvs, path_to_local, path_to_local_id, peel_hir_pat_refs, peel_n_hir_expr_refs, recurse_or_patterns,
    remove_blocks, strip_pat_refs,
};
use clippy_utils::{paths, search_same, SpanlessEq, SpanlessHash};
use core::array;
use core::iter::{once, ExactSizeIterator};
use if_chain::if_chain;
use rustc_ast::ast::{Attribute, LitKind};
use rustc_errors::Applicability;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::LangItem::{OptionNone, OptionSome};
use rustc_hir::{
    self as hir, Arm, BindingAnnotation, Block, BorrowKind, Expr, ExprKind, Guard, HirId, Local, MatchSource,
    Mutability, Node, Pat, PatKind, PathSegment, QPath, RangeEnd, TyKind,
};
use rustc_hir::{HirIdMap, HirIdSet};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{self, Ty, TyS, VariantDef};
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::source_map::{Span, Spanned};
use rustc_span::sym;
use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::iter;
use std::ops::Bound;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for matches with a single arm where an `if let`
    /// will usually suffice.
    ///
    /// ### Why is this bad?
    /// Just readability – `if let` nests less than a `match`.
    ///
    /// ### Example
    /// ```rust
    /// # fn bar(stool: &str) {}
    /// # let x = Some("abc");
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
    /// ### What it does
    /// Checks for matches with two arms where an `if let else` will
    /// usually suffice.
    ///
    /// ### Why is this bad?
    /// Just readability – `if let` nests less than a `match`.
    ///
    /// ### Known problems
    /// Personal style preferences may differ.
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks for matches where all arms match a reference,
    /// suggesting to remove the reference and deref the matched expression
    /// instead. It also checks for `if let &foo = bar` blocks.
    ///
    /// ### Why is this bad?
    /// It just makes the code less readable. That reference
    /// destructuring adds nothing to the code.
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks for matches where match expression is a `bool`. It
    /// suggests to replace the expression with an `if...else` block.
    ///
    /// ### Why is this bad?
    /// It makes the code less readable.
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks for overlapping match arms.
    ///
    /// ### Why is this bad?
    /// It is likely to be an error and if not, makes the code
    /// less obvious.
    ///
    /// ### Example
    /// ```rust
    /// let x = 5;
    /// match x {
    ///     1..=10 => println!("1 ... 10"),
    ///     5..=15 => println!("5 ... 15"),
    ///     _ => (),
    /// }
    /// ```
    pub MATCH_OVERLAPPING_ARM,
    style,
    "a `match` with overlapping arms"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for arm which matches all errors with `Err(_)`
    /// and take drastic actions like `panic!`.
    ///
    /// ### Why is this bad?
    /// It is generally a bad practice, similar to
    /// catching all exceptions in java with `catch(Exception)`
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks for match which is used to add a reference to an
    /// `Option` value.
    ///
    /// ### Why is this bad?
    /// Using `as_ref()` or `as_mut()` instead is shorter.
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks for wildcard enum matches using `_`.
    ///
    /// ### Why is this bad?
    /// New enum variants added by library updates can be missed.
    ///
    /// ### Known problems
    /// Suggested replacements may be incorrect if guards exhaustively cover some
    /// variants, and also may not use correct path to enum if it's not present in the current scope.
    ///
    /// ### Example
    /// ```rust
    /// # enum Foo { A(usize), B(usize) }
    /// # let x = Foo::B(1);
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
    /// ### What it does
    /// Checks for wildcard enum matches for a single variant.
    ///
    /// ### Why is this bad?
    /// New enum variants added by library updates can be missed.
    ///
    /// ### Known problems
    /// Suggested replacements may not use correct path to enum
    /// if it's not present in the current scope.
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks for wildcard pattern used with others patterns in same match arm.
    ///
    /// ### Why is this bad?
    /// Wildcard pattern already covers any other pattern as it will match anyway.
    /// It makes the code less readable, especially to spot wildcard pattern use in match arm.
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks for matches being used to destructure a single-variant enum
    /// or tuple struct where a `let` will suffice.
    ///
    /// ### Why is this bad?
    /// Just readability – `let` doesn't nest, whereas a `match` does.
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks for useless match that binds to only one value.
    ///
    /// ### Why is this bad?
    /// Readability and needless complexity.
    ///
    /// ### Known problems
    ///  Suggested replacements may be incorrect when `match`
    /// is actually binding temporary value, bringing a 'dropped while borrowed' error.
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks for unnecessary '..' pattern binding on struct when all fields are explicitly matched.
    ///
    /// ### Why is this bad?
    /// Correctness and readability. It's like having a wildcard pattern after
    /// matching all enum variants explicitly.
    ///
    /// ### Example
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
    /// ### What it does
    /// Lint for redundant pattern matching over `Result`, `Option`,
    /// `std::task::Poll` or `std::net::IpAddr`
    ///
    /// ### Why is this bad?
    /// It's more concise and clear to just use the proper
    /// utility function
    ///
    /// ### Known problems
    /// This will change the drop order for the matched type. Both `if let` and
    /// `while let` will drop the value at the end of the block, both `if` and `while` will drop the
    /// value before entering the block. For most types this change will not matter, but for a few
    /// types this will not be an acceptable change (e.g. locks). See the
    /// [reference](https://doc.rust-lang.org/reference/destructors.html#drop-scopes) for more about
    /// drop order.
    ///
    /// ### Example
    /// ```rust
    /// # use std::task::Poll;
    /// # use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    /// if let Ok(_) = Ok::<i32, i32>(42) {}
    /// if let Err(_) = Err::<i32, i32>(42) {}
    /// if let None = None::<()> {}
    /// if let Some(_) = Some(42) {}
    /// if let Poll::Pending = Poll::Pending::<()> {}
    /// if let Poll::Ready(_) = Poll::Ready(42) {}
    /// if let IpAddr::V4(_) = IpAddr::V4(Ipv4Addr::LOCALHOST) {}
    /// if let IpAddr::V6(_) = IpAddr::V6(Ipv6Addr::LOCALHOST) {}
    /// match Ok::<i32, i32>(42) {
    ///     Ok(_) => true,
    ///     Err(_) => false,
    /// };
    /// ```
    ///
    /// The more idiomatic use would be:
    ///
    /// ```rust
    /// # use std::task::Poll;
    /// # use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    /// if Ok::<i32, i32>(42).is_ok() {}
    /// if Err::<i32, i32>(42).is_err() {}
    /// if None::<()>.is_none() {}
    /// if Some(42).is_some() {}
    /// if Poll::Pending::<()>.is_pending() {}
    /// if Poll::Ready(42).is_ready() {}
    /// if IpAddr::V4(Ipv4Addr::LOCALHOST).is_ipv4() {}
    /// if IpAddr::V6(Ipv6Addr::LOCALHOST).is_ipv6() {}
    /// Ok::<i32, i32>(42).is_ok();
    /// ```
    pub REDUNDANT_PATTERN_MATCHING,
    style,
    "use the proper utility function avoiding an `if let`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `match`  or `if let` expressions producing a
    /// `bool` that could be written using `matches!`
    ///
    /// ### Why is this bad?
    /// Readability and needless complexity.
    ///
    /// ### Known problems
    /// This lint falsely triggers, if there are arms with
    /// `cfg` attributes that remove an arm evaluating to `false`.
    ///
    /// ### Example
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

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `match` with identical arm bodies.
    ///
    /// ### Why is this bad?
    /// This is probably a copy & paste error. If arm bodies
    /// are the same on purpose, you can factor them
    /// [using `|`](https://doc.rust-lang.org/book/patterns.html#multiple-patterns).
    ///
    /// ### Known problems
    /// False positive possible with order dependent `match`
    /// (see issue
    /// [#860](https://github.com/rust-lang/rust-clippy/issues/860)).
    ///
    /// ### Example
    /// ```rust,ignore
    /// match foo {
    ///     Bar => bar(),
    ///     Quz => quz(),
    ///     Baz => bar(), // <= oops
    /// }
    /// ```
    ///
    /// This should probably be
    /// ```rust,ignore
    /// match foo {
    ///     Bar => bar(),
    ///     Quz => quz(),
    ///     Baz => baz(), // <= fixed
    /// }
    /// ```
    ///
    /// or if the original code was not a typo:
    /// ```rust,ignore
    /// match foo {
    ///     Bar | Baz => bar(), // <= shows the intent better
    ///     Quz => quz(),
    /// }
    /// ```
    pub MATCH_SAME_ARMS,
    pedantic,
    "`match` with identical arm bodies"
}

#[derive(Default)]
pub struct Matches {
    msrv: Option<RustcVersion>,
    infallible_destructuring_match_linted: bool,
}

impl Matches {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        Self {
            msrv,
            ..Matches::default()
        }
    }
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
    MATCH_LIKE_MATCHES_MACRO,
    MATCH_SAME_ARMS,
]);

impl<'tcx> LateLintPass<'tcx> for Matches {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if in_external_macro(cx.sess(), expr.span) || in_macro(expr.span) {
            return;
        }

        redundant_pattern_match::check(cx, expr);

        if meets_msrv(self.msrv.as_ref(), &msrvs::MATCHES_MACRO) {
            if !check_match_like_matches(cx, expr) {
                lint_match_arms(cx, expr);
            }
        } else {
            lint_match_arms(cx, expr);
        }

        if let ExprKind::Match(ex, arms, MatchSource::Normal) = expr.kind {
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
        if let ExprKind::Match(ex, arms, _) = expr.kind {
            check_match_ref_pats(cx, ex, arms.iter().map(|el| el.pat), expr);
        }
        if let Some(higher::IfLet { let_pat, let_expr, .. }) = higher::IfLet::hir(cx, expr) {
            check_match_ref_pats(cx, let_expr, once(let_pat), expr);
        }
    }

    fn check_local(&mut self, cx: &LateContext<'tcx>, local: &'tcx Local<'_>) {
        if_chain! {
            if !in_external_macro(cx.sess(), local.span);
            if !in_macro(local.span);
            if let Some(expr) = local.init;
            if let ExprKind::Match(target, arms, MatchSource::Normal) = expr.kind;
            if arms.len() == 1 && arms[0].guard.is_none();
            if let PatKind::TupleStruct(
                QPath::Resolved(None, variant_name), args, _) = arms[0].pat.kind;
            if args.len() == 1;
            if let PatKind::Binding(_, arg, ..) = strip_pat_refs(&args[0]).kind;
            let body = remove_blocks(arms[0].body);
            if path_to_local_id(body, arg);

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
            if let PatKind::Struct(QPath::Resolved(_, path), fields, true) = pat.kind;
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

    extract_msrv_attr!(LateContext);
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
            }
            // block with 2+ statements or 1 expr and 1+ statement
            Some(els)
        } else {
            // not a block, don't lint
            return;
        };

        let ty = cx.typeck_results().expr_ty(ex);
        if *ty.kind() != ty::Bool || is_lint_allowed(cx, MATCH_BOOL, ex.hir_id) {
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
    if is_wild(arms[1].pat) {
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

    let (pat, pat_ref_count) = peel_hir_pat_refs(arms[0].pat);
    let (msg, sugg) = if_chain! {
        if let PatKind::Path(_) | PatKind::Lit(_) = pat.kind;
        let (ty, ty_ref_count) = peel_mid_ty_refs(cx.typeck_results().expr_ty(ex));
        if let Some(spe_trait_id) = cx.tcx.lang_items().structural_peq_trait();
        if let Some(pe_trait_id) = cx.tcx.lang_items().eq_trait();
        if ty.is_integral() || ty.is_char() || ty.is_str()
            || (implements_trait(cx, ty, spe_trait_id, &[])
                && implements_trait(cx, ty, pe_trait_id, &[ty.into()]));
        then {
            // scrutinee derives PartialEq and the pattern is a constant.
            let pat_ref_count = match pat.kind {
                // string literals are already a reference.
                PatKind::Lit(Expr { kind: ExprKind::Lit(lit), .. }) if lit.node.is_str() => pat_ref_count + 1,
                _ => pat_ref_count,
            };
            // References are only implicitly added to the pattern, so no overflow here.
            // e.g. will work: match &Some(_) { Some(_) => () }
            // will not: match Some(_) { &Some(_) => () }
            let ref_count_diff = ty_ref_count - pat_ref_count;

            // Try to remove address of expressions first.
            let (ex, removed) = peel_n_hir_expr_refs(ex, ref_count_diff);
            let ref_count_diff = ref_count_diff - removed;

            let msg = "you seem to be trying to use `match` for an equality check. Consider using `if`";
            let sugg = format!(
                "if {} == {}{} {}{}",
                snippet(cx, ex.span, ".."),
                // PartialEq for different reference counts may not exist.
                "&".repeat(ref_count_diff),
                snippet(cx, arms[0].pat.span, ".."),
                expr_block(cx, arms[0].body, None, "..", Some(expr.span)),
                els_str,
            );
            (msg, sugg)
        } else {
            let msg = "you seem to be trying to use `match` for destructuring a single pattern. Consider using `if let`";
            let sugg = format!(
                "if let {} = {} {}{}",
                snippet(cx, arms[0].pat.span, ".."),
                snippet(cx, ex.span, ".."),
                expr_block(cx, arms[0].body, None, "..", Some(expr.span)),
                els_str,
            );
            (msg, sugg)
        }
    };

    span_lint_and_sugg(
        cx,
        lint,
        expr.span,
        msg,
        "try this",
        sugg,
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
        PatKind::TupleStruct(ref path, inner, _) => {
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
                    let exprs = if let PatKind::Lit(arm_bool) = arms[0].pat.kind {
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

fn check_wild_err_arm<'tcx>(cx: &LateContext<'tcx>, ex: &Expr<'tcx>, arms: &[Arm<'tcx>]) {
    let ex_ty = cx.typeck_results().expr_ty(ex).peel_refs();
    if is_type_diagnostic_item(cx, ex_ty, sym::Result) {
        for arm in arms {
            if let PatKind::TupleStruct(ref path, inner, _) = arm.pat.kind {
                let path_str = rustc_hir_pretty::to_string(rustc_hir_pretty::NO_ANN, |s| s.print_qpath(path, false));
                if path_str == "Err" {
                    let mut matching_wild = inner.iter().any(is_wild);
                    let mut ident_bind_name = String::from("_");
                    if !matching_wild {
                        // Looking for unused bindings (i.e.: `_e`)
                        for pat in inner.iter() {
                            if let PatKind::Binding(_, id, ident, None) = pat.kind {
                                if ident.as_str().starts_with('_') && !is_local_used(cx, arm.body, id) {
                                    ident_bind_name = (&ident.name.as_str()).to_string();
                                    matching_wild = true;
                                }
                            }
                        }
                    }
                    if_chain! {
                        if matching_wild;
                        if let ExprKind::Block(block, _) = arm.body.kind;
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

enum CommonPrefixSearcher<'a> {
    None,
    Path(&'a [PathSegment<'a>]),
    Mixed,
}
impl CommonPrefixSearcher<'a> {
    fn with_path(&mut self, path: &'a [PathSegment<'a>]) {
        match path {
            [path @ .., _] => self.with_prefix(path),
            [] => (),
        }
    }

    fn with_prefix(&mut self, path: &'a [PathSegment<'a>]) {
        match self {
            Self::None => *self = Self::Path(path),
            Self::Path(self_path)
                if path
                    .iter()
                    .map(|p| p.ident.name)
                    .eq(self_path.iter().map(|p| p.ident.name)) => {},
            Self::Path(_) => *self = Self::Mixed,
            Self::Mixed => (),
        }
    }
}

fn is_hidden(cx: &LateContext<'_>, variant_def: &VariantDef) -> bool {
    let attrs = cx.tcx.get_attrs(variant_def.def_id);
    clippy_utils::attrs::is_doc_hidden(attrs) || clippy_utils::attrs::is_unstable(attrs)
}

#[allow(clippy::too_many_lines)]
fn check_wild_enum_match(cx: &LateContext<'_>, ex: &Expr<'_>, arms: &[Arm<'_>]) {
    let ty = cx.typeck_results().expr_ty(ex).peel_refs();
    let adt_def = match ty.kind() {
        ty::Adt(adt_def, _)
            if adt_def.is_enum()
                && !(is_type_diagnostic_item(cx, ty, sym::Option) || is_type_diagnostic_item(cx, ty, sym::Result)) =>
        {
            adt_def
        },
        _ => return,
    };

    // First pass - check for violation, but don't do much book-keeping because this is hopefully
    // the uncommon case, and the book-keeping is slightly expensive.
    let mut wildcard_span = None;
    let mut wildcard_ident = None;
    let mut has_non_wild = false;
    for arm in arms {
        match peel_hir_pat_refs(arm.pat).0.kind {
            PatKind::Wild => wildcard_span = Some(arm.pat.span),
            PatKind::Binding(_, _, ident, None) => {
                wildcard_span = Some(arm.pat.span);
                wildcard_ident = Some(ident);
            },
            _ => has_non_wild = true,
        }
    }
    let wildcard_span = match wildcard_span {
        Some(x) if has_non_wild => x,
        _ => return,
    };

    // Accumulate the variants which should be put in place of the wildcard because they're not
    // already covered.
    let has_hidden = adt_def.variants.iter().any(|x| is_hidden(cx, x));
    let mut missing_variants: Vec<_> = adt_def.variants.iter().filter(|x| !is_hidden(cx, x)).collect();

    let mut path_prefix = CommonPrefixSearcher::None;
    for arm in arms {
        // Guards mean that this case probably isn't exhaustively covered. Technically
        // this is incorrect, as we should really check whether each variant is exhaustively
        // covered by the set of guards that cover it, but that's really hard to do.
        recurse_or_patterns(arm.pat, |pat| {
            let path = match &peel_hir_pat_refs(pat).0.kind {
                PatKind::Path(path) => {
                    #[allow(clippy::match_same_arms)]
                    let id = match cx.qpath_res(path, pat.hir_id) {
                        Res::Def(DefKind::Const | DefKind::ConstParam | DefKind::AnonConst, _) => return,
                        Res::Def(_, id) => id,
                        _ => return,
                    };
                    if arm.guard.is_none() {
                        missing_variants.retain(|e| e.ctor_def_id != Some(id));
                    }
                    path
                },
                PatKind::TupleStruct(path, patterns, ..) => {
                    if let Some(id) = cx.qpath_res(path, pat.hir_id).opt_def_id() {
                        if arm.guard.is_none() && patterns.iter().all(|p| !is_refutable(cx, p)) {
                            missing_variants.retain(|e| e.ctor_def_id != Some(id));
                        }
                    }
                    path
                },
                PatKind::Struct(path, patterns, ..) => {
                    if let Some(id) = cx.qpath_res(path, pat.hir_id).opt_def_id() {
                        if arm.guard.is_none() && patterns.iter().all(|p| !is_refutable(cx, p.pat)) {
                            missing_variants.retain(|e| e.def_id != id);
                        }
                    }
                    path
                },
                _ => return,
            };
            match path {
                QPath::Resolved(_, path) => path_prefix.with_path(path.segments),
                QPath::TypeRelative(
                    hir::Ty {
                        kind: TyKind::Path(QPath::Resolved(_, path)),
                        ..
                    },
                    _,
                ) => path_prefix.with_prefix(path.segments),
                _ => (),
            }
        });
    }

    let format_suggestion = |variant: &VariantDef| {
        format!(
            "{}{}{}{}",
            if let Some(ident) = wildcard_ident {
                format!("{} @ ", ident.name)
            } else {
                String::new()
            },
            if let CommonPrefixSearcher::Path(path_prefix) = path_prefix {
                let mut s = String::new();
                for seg in path_prefix {
                    s.push_str(&seg.ident.as_str());
                    s.push_str("::");
                }
                s
            } else {
                let mut s = cx.tcx.def_path_str(adt_def.did);
                s.push_str("::");
                s
            },
            variant.ident.name,
            match variant.ctor_kind {
                CtorKind::Fn if variant.fields.len() == 1 => "(_)",
                CtorKind::Fn => "(..)",
                CtorKind::Const => "",
                CtorKind::Fictive => "{ .. }",
            }
        )
    };

    match missing_variants.as_slice() {
        [] => (),
        [x] if !adt_def.is_variant_list_non_exhaustive() && !has_hidden => span_lint_and_sugg(
            cx,
            MATCH_WILDCARD_FOR_SINGLE_VARIANTS,
            wildcard_span,
            "wildcard matches only a single variant and will also match any future added variants",
            "try this",
            format_suggestion(x),
            Applicability::MaybeIncorrect,
        ),
        variants => {
            let mut suggestions: Vec<_> = variants.iter().copied().map(format_suggestion).collect();
            let message = if adt_def.is_variant_list_non_exhaustive() || has_hidden {
                suggestions.push("_".into());
                "wildcard matches known variants and will also match future added variants"
            } else {
                "wildcard match will also match any future added variants"
            };

            span_lint_and_sugg(
                cx,
                WILDCARD_ENUM_MATCH_ARM,
                wildcard_span,
                message,
                "try this",
                suggestions.join(" | "),
                Applicability::MaybeIncorrect,
            );
        },
    };
}

// If the block contains only a `panic!` macro (as expression or statement)
fn is_panic_block(block: &Block<'_>) -> bool {
    match (&block.expr, block.stmts.len(), block.stmts.first()) {
        (&Some(exp), 0, _) => is_expn_of(exp.span, "panic").is_some() && is_expn_of(exp.span, "unreachable").is_none(),
        (&None, 1, Some(stmt)) => {
            is_expn_of(stmt.span, "panic").is_some() && is_expn_of(stmt.span, "unreachable").is_none()
        },
        _ => false,
    }
}

fn check_match_ref_pats<'a, 'b, I>(cx: &LateContext<'_>, ex: &Expr<'_>, pats: I, expr: &Expr<'_>)
where
    'b: 'a,
    I: Clone + Iterator<Item = &'a Pat<'b>>,
{
    if !has_only_ref_pats(pats.clone()) {
        return;
    }

    let (first_sugg, msg, title);
    let span = ex.span.source_callsite();
    if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, inner) = ex.kind {
        first_sugg = once((span, Sugg::hir_with_macro_callsite(cx, inner, "..").to_string()));
        msg = "try";
        title = "you don't need to add `&` to both the expression and the patterns";
    } else {
        first_sugg = once((span, Sugg::hir_with_macro_callsite(cx, ex, "..").deref().to_string()));
        msg = "instead of prefixing all patterns with `&`, you can dereference the expression";
        title = "you don't need to add `&` to all patterns";
    }

    let remaining_suggs = pats.filter_map(|pat| {
        if let PatKind::Ref(refp, _) = pat.kind {
            Some((pat.span, snippet(cx, refp.span, "..").to_string()))
        } else {
            None
        }
    });

    span_lint_and_then(cx, MATCH_REF_PATS, expr.span, title, |diag| {
        if !expr.span.from_expansion() {
            multispan_sugg(diag, msg, first_sugg.chain(remaining_suggs));
        }
    });
}

fn check_match_as_ref(cx: &LateContext<'_>, ex: &Expr<'_>, arms: &[Arm<'_>], expr: &Expr<'_>) {
    if arms.len() == 2 && arms[0].guard.is_none() && arms[1].guard.is_none() {
        let arm_ref: Option<BindingAnnotation> = if is_none_arm(cx, &arms[0]) {
            is_ref_some_arm(cx, &arms[1])
        } else if is_none_arm(cx, &arms[1]) {
            is_ref_some_arm(cx, &arms[0])
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
            );
        }
    }
}

fn check_wild_in_or_pats(cx: &LateContext<'_>, arms: &[Arm<'_>]) {
    for arm in arms {
        if let PatKind::Or(fields) = arm.pat.kind {
            // look for multiple fields in this arm that contains at least one Wild pattern
            if fields.len() > 1 && fields.iter().any(is_wild) {
                span_lint_and_help(
                    cx,
                    WILDCARD_IN_OR_PATTERNS,
                    arm.pat.span,
                    "wildcard pattern covers any other pattern as it will match anyway",
                    None,
                    "consider handling `_` separately",
                );
            }
        }
    }
}

/// Lint a `match` or `if let .. { .. } else { .. }` expr that could be replaced by `matches!`
fn check_match_like_matches<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> bool {
    if let Some(higher::IfLet {
        let_pat,
        let_expr,
        if_then,
        if_else: Some(if_else),
    }) = higher::IfLet::hir(cx, expr)
    {
        return find_matches_sugg(
            cx,
            let_expr,
            array::IntoIter::new([(&[][..], Some(let_pat), if_then, None), (&[][..], None, if_else, None)]),
            expr,
            true,
        );
    }

    if let ExprKind::Match(scrut, arms, MatchSource::Normal) = expr.kind {
        return find_matches_sugg(
            cx,
            scrut,
            arms.iter().map(|arm| {
                (
                    cx.tcx.hir().attrs(arm.hir_id),
                    Some(arm.pat),
                    arm.body,
                    arm.guard.as_ref(),
                )
            }),
            expr,
            false,
        );
    }

    false
}

/// Lint a `match` or `if let` for replacement by `matches!`
fn find_matches_sugg<'a, 'b, I>(
    cx: &LateContext<'_>,
    ex: &Expr<'_>,
    mut iter: I,
    expr: &Expr<'_>,
    is_if_let: bool,
) -> bool
where
    'b: 'a,
    I: Clone
        + DoubleEndedIterator
        + ExactSizeIterator
        + Iterator<
            Item = (
                &'a [Attribute],
                Option<&'a Pat<'b>>,
                &'a Expr<'b>,
                Option<&'a Guard<'b>>,
            ),
        >,
{
    if_chain! {
        if iter.len() >= 2;
        if cx.typeck_results().expr_ty(expr).is_bool();
        if let Some((_, last_pat_opt, last_expr, _)) = iter.next_back();
        let iter_without_last = iter.clone();
        if let Some((first_attrs, _, first_expr, first_guard)) = iter.next();
        if let Some(b0) = find_bool_lit(&first_expr.kind, is_if_let);
        if let Some(b1) = find_bool_lit(&last_expr.kind, is_if_let);
        if b0 != b1;
        if first_guard.is_none() || iter.len() == 0;
        if first_attrs.is_empty();
        if iter
            .all(|arm| {
                find_bool_lit(&arm.2.kind, is_if_let).map_or(false, |b| b == b0) && arm.3.is_none() && arm.0.is_empty()
            });
        then {
            if let Some(last_pat) = last_pat_opt {
                if !is_wild(last_pat) {
                    return false;
                }
            }

            // The suggestion may be incorrect, because some arms can have `cfg` attributes
            // evaluated into `false` and so such arms will be stripped before.
            let mut applicability = Applicability::MaybeIncorrect;
            let pat = {
                use itertools::Itertools as _;
                iter_without_last
                    .filter_map(|arm| {
                        let pat_span = arm.1?.span;
                        Some(snippet_with_applicability(cx, pat_span, "..", &mut applicability))
                    })
                    .join(" | ")
            };
            let pat_and_guard = if let Some(Guard::If(g)) = first_guard {
                format!("{} if {}", pat, snippet_with_applicability(cx, g.span, "..", &mut applicability))
            } else {
                pat
            };

            // strip potential borrows (#6503), but only if the type is a reference
            let mut ex_new = ex;
            if let ExprKind::AddrOf(BorrowKind::Ref, .., ex_inner) = ex.kind {
                if let ty::Ref(..) = cx.typeck_results().expr_ty(ex_inner).kind() {
                    ex_new = ex_inner;
                }
            };
            span_lint_and_sugg(
                cx,
                MATCH_LIKE_MATCHES_MACRO,
                expr.span,
                &format!("{} expression looks like `matches!` macro", if is_if_let { "if let .. else" } else { "match" }),
                "try this",
                format!(
                    "{}matches!({}, {})",
                    if b0 { "" } else { "!" },
                    snippet_with_applicability(cx, ex_new.span, "..", &mut applicability),
                    pat_and_guard,
                ),
                applicability,
            );
            true
        } else {
            false
        }
    }
}

/// Extract a `bool` or `{ bool }`
fn find_bool_lit(ex: &ExprKind<'_>, is_if_let: bool) -> Option<bool> {
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
        ) if is_if_let => {
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

#[allow(clippy::too_many_lines)]
fn check_match_single_binding<'a>(cx: &LateContext<'a>, ex: &Expr<'a>, arms: &[Arm<'_>], expr: &Expr<'_>) {
    if in_macro(expr.span) || arms.len() != 1 || is_refutable(cx, arms[0].pat) {
        return;
    }

    // HACK:
    // This is a hack to deal with arms that are excluded by macros like `#[cfg]`. It is only used here
    // to prevent false positives as there is currently no better way to detect if code was excluded by
    // a macro. See PR #6435
    if_chain! {
        if let Some(match_snippet) = snippet_opt(cx, expr.span);
        if let Some(arm_snippet) = snippet_opt(cx, arms[0].span);
        if let Some(ex_snippet) = snippet_opt(cx, ex.span);
        let rest_snippet = match_snippet.replace(&arm_snippet, "").replace(&ex_snippet, "");
        if rest_snippet.contains("=>");
        then {
            // The code it self contains another thick arrow "=>"
            // -> Either another arm or a comment
            return;
        }
    }

    let matched_vars = ex.span;
    let bind_names = arms[0].pat.span;
    let match_body = remove_blocks(arms[0].body);
    let mut snippet_body = if match_body.span.from_expansion() {
        Sugg::hir_with_macro_callsite(cx, match_body, "..").to_string()
    } else {
        snippet_block(cx, match_body.span, "..", Some(expr.span)).to_string()
    };

    // Do we need to add ';' to suggestion ?
    match match_body.kind {
        ExprKind::Block(block, _) => {
            // macro + expr_ty(body) == ()
            if block.span.from_expansion() && cx.typeck_results().expr_ty(match_body).is_unit() {
                snippet_body.push(';');
            }
        },
        _ => {
            // expr_ty(body) == ()
            if cx.typeck_results().expr_ty(match_body).is_unit() {
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
                }
                // If the parent is already an arm, and the body is another match statement,
                // we need curly braces around suggestion
                let parent_node_id = cx.tcx.hir().get_parent_node(expr.hir_id);
                if let Node::Arm(arm) = &cx.tcx.hir().get(parent_node_id) {
                    if let ExprKind::Match(..) = arm.body.kind {
                        cbrace_end = format!("\n{}}}", indent);
                        // Fix body indent due to the match
                        indent = " ".repeat(indent_of(cx, bind_names).unwrap_or(0));
                        cbrace_start = format!("{{\n{}", indent);
                    }
                }
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
            if ex.can_have_side_effects() {
                let indent = " ".repeat(indent_of(cx, expr.span).unwrap_or(0));
                let sugg = format!(
                    "{};\n{}{}",
                    snippet_with_applicability(cx, ex.span, "..", &mut applicability),
                    indent,
                    snippet_body
                );
                span_lint_and_sugg(
                    cx,
                    MATCH_SINGLE_BINDING,
                    expr.span,
                    "this match could be replaced by its scrutinee and body",
                    "consider using the scrutinee and body instead",
                    sugg,
                    applicability,
                );
            } else {
                span_lint_and_sugg(
                    cx,
                    MATCH_SINGLE_BINDING,
                    expr.span,
                    "this match could be replaced by its body itself",
                    "consider using the match body instead",
                    snippet_body,
                    Applicability::MachineApplicable,
                );
            }
        },
        _ => (),
    }
}

/// Returns true if the `ex` match expression is in a local (`let`) statement
fn opt_parent_let<'a>(cx: &LateContext<'a>, ex: &Expr<'a>) -> Option<&'a Local<'a>> {
    let map = &cx.tcx.hir();
    if_chain! {
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
        .filter_map(|arm| {
            if let Arm { pat, guard: None, .. } = *arm {
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

                if let PatKind::Lit(value) = pat.kind {
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

// Checks if arm has the form `None => None`
fn is_none_arm(cx: &LateContext<'_>, arm: &Arm<'_>) -> bool {
    matches!(arm.pat.kind, PatKind::Path(ref qpath) if is_lang_ctor(cx, qpath, OptionNone))
}

// Checks if arm has the form `Some(ref v) => Some(v)` (checks for `ref` and `ref mut`)
fn is_ref_some_arm(cx: &LateContext<'_>, arm: &Arm<'_>) -> Option<BindingAnnotation> {
    if_chain! {
        if let PatKind::TupleStruct(ref qpath, [first_pat, ..], _) = arm.pat.kind;
        if is_lang_ctor(cx, qpath, OptionSome);
        if let PatKind::Binding(rb, .., ident, _) = first_pat.kind;
        if rb == BindingAnnotation::Ref || rb == BindingAnnotation::RefMut;
        if let ExprKind::Call(e, args) = remove_blocks(arm.body).kind;
        if let ExprKind::Path(ref some_path) = e.kind;
        if is_lang_ctor(cx, some_path, OptionSome) && args.len() == 1;
        if let ExprKind::Path(QPath::Resolved(_, path2)) = args[0].kind;
        if path2.segments.len() == 1 && ident.name == path2.segments[0].ident.name;
        then {
            return Some(rb)
        }
    }
    None
}

fn has_only_ref_pats<'a, 'b, I>(pats: I) -> bool
where
    'b: 'a,
    I: Iterator<Item = &'a Pat<'b>>,
{
    let mut at_least_one_is_true = false;
    for opt in pats.map(|pat| match pat.kind {
        PatKind::Ref(..) => Some(true), // &-patterns
        PatKind::Wild => Some(false),   // an "anything" wildcard is also fine
        _ => None,                      // any other pattern is not fine
    }) {
        if let Some(inner) = opt {
            if inner {
                at_least_one_is_true = true;
            }
        } else {
            return false;
        }
    }
    at_least_one_is_true
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

    for (a, b) in iter::zip(&values, values.iter().skip(1)) {
        match (a, b) {
            (&Kind::Start(_, ra), &Kind::End(_, rb)) => {
                if ra.node != rb.node {
                    return Some((ra, rb));
                }
            },
            (&Kind::End(a, _), &Kind::Start(b, _)) if a != Bound::Included(b) => (),
            _ => {
                // skip if the range `a` is completely included into the range `b`
                if let Ordering::Equal | Ordering::Less = a.cmp(b) {
                    let kind_a = Kind::End(a.range().node.1, a.range());
                    let kind_b = Kind::End(b.range().node.1, b.range());
                    if let Ordering::Equal | Ordering::Greater = kind_a.cmp(&kind_b) {
                        return None;
                    }
                }
                return Some((a.range(), b.range()));
            },
        }
    }

    None
}

mod redundant_pattern_match {
    use super::REDUNDANT_PATTERN_MATCHING;
    use clippy_utils::diagnostics::span_lint_and_then;
    use clippy_utils::higher;
    use clippy_utils::source::{snippet, snippet_with_applicability};
    use clippy_utils::ty::{implements_trait, is_type_diagnostic_item, is_type_lang_item, match_type};
    use clippy_utils::{is_lang_ctor, is_qpath_def_path, is_trait_method, paths};
    use if_chain::if_chain;
    use rustc_ast::ast::LitKind;
    use rustc_data_structures::fx::FxHashSet;
    use rustc_errors::Applicability;
    use rustc_hir::LangItem::{OptionNone, OptionSome, PollPending, PollReady, ResultErr, ResultOk};
    use rustc_hir::{
        intravisit::{walk_expr, ErasedMap, NestedVisitorMap, Visitor},
        Arm, Block, Expr, ExprKind, LangItem, MatchSource, Node, Pat, PatKind, QPath,
    };
    use rustc_lint::LateContext;
    use rustc_middle::ty::{self, subst::GenericArgKind, Ty};
    use rustc_span::sym;

    pub fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let Some(higher::IfLet {
            if_else,
            let_pat,
            let_expr,
            ..
        }) = higher::IfLet::hir(cx, expr)
        {
            find_sugg_for_if_let(cx, expr, let_pat, let_expr, "if", if_else.is_some());
        }
        if let ExprKind::Match(op, arms, MatchSource::Normal) = &expr.kind {
            find_sugg_for_match(cx, expr, op, arms);
        }
        if let Some(higher::WhileLet { let_pat, let_expr, .. }) = higher::WhileLet::hir(expr) {
            find_sugg_for_if_let(cx, expr, let_pat, let_expr, "while", false);
        }
    }

    /// Checks if the drop order for a type matters. Some std types implement drop solely to
    /// deallocate memory. For these types, and composites containing them, changing the drop order
    /// won't result in any observable side effects.
    fn type_needs_ordered_drop(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> bool {
        type_needs_ordered_drop_inner(cx, ty, &mut FxHashSet::default())
    }

    fn type_needs_ordered_drop_inner(cx: &LateContext<'tcx>, ty: Ty<'tcx>, seen: &mut FxHashSet<Ty<'tcx>>) -> bool {
        if !seen.insert(ty) {
            return false;
        }
        if !ty.needs_drop(cx.tcx, cx.param_env) {
            false
        } else if !cx
            .tcx
            .lang_items()
            .drop_trait()
            .map_or(false, |id| implements_trait(cx, ty, id, &[]))
        {
            // This type doesn't implement drop, so no side effects here.
            // Check if any component type has any.
            match ty.kind() {
                ty::Tuple(_) => ty.tuple_fields().any(|ty| type_needs_ordered_drop_inner(cx, ty, seen)),
                ty::Array(ty, _) => type_needs_ordered_drop_inner(cx, ty, seen),
                ty::Adt(adt, subs) => adt
                    .all_fields()
                    .map(|f| f.ty(cx.tcx, subs))
                    .any(|ty| type_needs_ordered_drop_inner(cx, ty, seen)),
                _ => true,
            }
        }
        // Check for std types which implement drop, but only for memory allocation.
        else if is_type_diagnostic_item(cx, ty, sym::Vec)
            || is_type_lang_item(cx, ty, LangItem::OwnedBox)
            || is_type_diagnostic_item(cx, ty, sym::Rc)
            || is_type_diagnostic_item(cx, ty, sym::Arc)
            || is_type_diagnostic_item(cx, ty, sym::cstring_type)
            || is_type_diagnostic_item(cx, ty, sym::BTreeMap)
            || is_type_diagnostic_item(cx, ty, sym::LinkedList)
            || match_type(cx, ty, &paths::WEAK_RC)
            || match_type(cx, ty, &paths::WEAK_ARC)
        {
            // Check all of the generic arguments.
            if let ty::Adt(_, subs) = ty.kind() {
                subs.types().any(|ty| type_needs_ordered_drop_inner(cx, ty, seen))
            } else {
                true
            }
        } else {
            true
        }
    }

    // Extract the generic arguments out of a type
    fn try_get_generic_ty(ty: Ty<'_>, index: usize) -> Option<Ty<'_>> {
        if_chain! {
            if let ty::Adt(_, subs) = ty.kind();
            if let Some(sub) = subs.get(index);
            if let GenericArgKind::Type(sub_ty) = sub.unpack();
            then {
                Some(sub_ty)
            } else {
                None
            }
        }
    }

    // Checks if there are any temporaries created in the given expression for which drop order
    // matters.
    fn temporaries_need_ordered_drop(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
        struct V<'a, 'tcx> {
            cx: &'a LateContext<'tcx>,
            res: bool,
        }
        impl<'a, 'tcx> Visitor<'tcx> for V<'a, 'tcx> {
            type Map = ErasedMap<'tcx>;
            fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
                NestedVisitorMap::None
            }

            fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
                match expr.kind {
                    // Taking the reference of a value leaves a temporary
                    // e.g. In `&String::new()` the string is a temporary value.
                    // Remaining fields are temporary values
                    // e.g. In `(String::new(), 0).1` the string is a temporary value.
                    ExprKind::AddrOf(_, _, expr) | ExprKind::Field(expr, _) => {
                        if !matches!(expr.kind, ExprKind::Path(_)) {
                            if type_needs_ordered_drop(self.cx, self.cx.typeck_results().expr_ty(expr)) {
                                self.res = true;
                            } else {
                                self.visit_expr(expr);
                            }
                        }
                    },
                    // the base type is alway taken by reference.
                    // e.g. In `(vec![0])[0]` the vector is a temporary value.
                    ExprKind::Index(base, index) => {
                        if !matches!(base.kind, ExprKind::Path(_)) {
                            if type_needs_ordered_drop(self.cx, self.cx.typeck_results().expr_ty(base)) {
                                self.res = true;
                            } else {
                                self.visit_expr(base);
                            }
                        }
                        self.visit_expr(index);
                    },
                    // Method calls can take self by reference.
                    // e.g. In `String::new().len()` the string is a temporary value.
                    ExprKind::MethodCall(_, _, [self_arg, args @ ..], _) => {
                        if !matches!(self_arg.kind, ExprKind::Path(_)) {
                            let self_by_ref = self
                                .cx
                                .typeck_results()
                                .type_dependent_def_id(expr.hir_id)
                                .map_or(false, |id| self.cx.tcx.fn_sig(id).skip_binder().inputs()[0].is_ref());
                            if self_by_ref
                                && type_needs_ordered_drop(self.cx, self.cx.typeck_results().expr_ty(self_arg))
                            {
                                self.res = true;
                            } else {
                                self.visit_expr(self_arg);
                            }
                        }
                        args.iter().for_each(|arg| self.visit_expr(arg));
                    },
                    // Either explicitly drops values, or changes control flow.
                    ExprKind::DropTemps(_)
                    | ExprKind::Ret(_)
                    | ExprKind::Break(..)
                    | ExprKind::Yield(..)
                    | ExprKind::Block(Block { expr: None, .. }, _)
                    | ExprKind::Loop(..) => (),

                    // Only consider the final expression.
                    ExprKind::Block(Block { expr: Some(expr), .. }, _) => self.visit_expr(expr),

                    _ => walk_expr(self, expr),
                }
            }
        }

        let mut v = V { cx, res: false };
        v.visit_expr(expr);
        v.res
    }

    fn find_sugg_for_if_let<'tcx>(
        cx: &LateContext<'tcx>,
        expr: &'tcx Expr<'_>,
        let_pat: &Pat<'_>,
        let_expr: &'tcx Expr<'_>,
        keyword: &'static str,
        has_else: bool,
    ) {
        // also look inside refs
        let mut kind = &let_pat.kind;
        // if we have &None for example, peel it so we can detect "if let None = x"
        if let PatKind::Ref(inner, _mutability) = kind {
            kind = &inner.kind;
        }
        let op_ty = cx.typeck_results().expr_ty(let_expr);
        // Determine which function should be used, and the type contained by the corresponding
        // variant.
        let (good_method, inner_ty) = match kind {
            PatKind::TupleStruct(ref path, [sub_pat], _) => {
                if let PatKind::Wild = sub_pat.kind {
                    if is_lang_ctor(cx, path, ResultOk) {
                        ("is_ok()", try_get_generic_ty(op_ty, 0).unwrap_or(op_ty))
                    } else if is_lang_ctor(cx, path, ResultErr) {
                        ("is_err()", try_get_generic_ty(op_ty, 1).unwrap_or(op_ty))
                    } else if is_lang_ctor(cx, path, OptionSome) {
                        ("is_some()", op_ty)
                    } else if is_lang_ctor(cx, path, PollReady) {
                        ("is_ready()", op_ty)
                    } else if is_qpath_def_path(cx, path, sub_pat.hir_id, &paths::IPADDR_V4) {
                        ("is_ipv4()", op_ty)
                    } else if is_qpath_def_path(cx, path, sub_pat.hir_id, &paths::IPADDR_V6) {
                        ("is_ipv6()", op_ty)
                    } else {
                        return;
                    }
                } else {
                    return;
                }
            },
            PatKind::Path(ref path) => {
                let method = if is_lang_ctor(cx, path, OptionNone) {
                    "is_none()"
                } else if is_lang_ctor(cx, path, PollPending) {
                    "is_pending()"
                } else {
                    return;
                };
                // `None` and `Pending` don't have an inner type.
                (method, cx.tcx.types.unit)
            },
            _ => return,
        };

        // If this is the last expression in a block or there is an else clause then the whole
        // type needs to be considered, not just the inner type of the branch being matched on.
        // Note the last expression in a block is dropped after all local bindings.
        let check_ty = if has_else
            || (keyword == "if" && matches!(cx.tcx.hir().parent_iter(expr.hir_id).next(), Some((_, Node::Block(..)))))
        {
            op_ty
        } else {
            inner_ty
        };

        // All temporaries created in the scrutinee expression are dropped at the same time as the
        // scrutinee would be, so they have to be considered as well.
        // e.g. in `if let Some(x) = foo.lock().unwrap().baz.as_ref() { .. }` the lock will be held
        // for the duration if body.
        let needs_drop = type_needs_ordered_drop(cx, check_ty) || temporaries_need_ordered_drop(cx, let_expr);

        // check that `while_let_on_iterator` lint does not trigger
        if_chain! {
            if keyword == "while";
            if let ExprKind::MethodCall(method_path, _, _, _) = let_expr.kind;
            if method_path.ident.name == sym::next;
            if is_trait_method(cx, let_expr, sym::Iterator);
            then {
                return;
            }
        }

        let result_expr = match &let_expr.kind {
            ExprKind::AddrOf(_, _, borrowed) => borrowed,
            _ => let_expr,
        };
        span_lint_and_then(
            cx,
            REDUNDANT_PATTERN_MATCHING,
            let_pat.span,
            &format!("redundant pattern matching, consider using `{}`", good_method),
            |diag| {
                // if/while let ... = ... { ... }
                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                let expr_span = expr.span;

                // if/while let ... = ... { ... }
                //                 ^^^
                let op_span = result_expr.span.source_callsite();

                // if/while let ... = ... { ... }
                // ^^^^^^^^^^^^^^^^^^^
                let span = expr_span.until(op_span.shrink_to_hi());

                let mut app = if needs_drop {
                    Applicability::MaybeIncorrect
                } else {
                    Applicability::MachineApplicable
                };
                let sugg = snippet_with_applicability(cx, op_span, "_", &mut app);

                diag.span_suggestion(span, "try this", format!("{} {}.{}", keyword, sugg, good_method), app);

                if needs_drop {
                    diag.note("this will change drop order of the result, as well as all temporaries");
                    diag.note("add `#[allow(clippy::redundant_pattern_matching)]` if this is important");
                }
            },
        );
    }

    fn find_sugg_for_match<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, op: &Expr<'_>, arms: &[Arm<'_>]) {
        if arms.len() == 2 {
            let node_pair = (&arms[0].pat.kind, &arms[1].pat.kind);

            let found_good_method = match node_pair {
                (
                    PatKind::TupleStruct(ref path_left, patterns_left, _),
                    PatKind::TupleStruct(ref path_right, patterns_right, _),
                ) if patterns_left.len() == 1 && patterns_right.len() == 1 => {
                    if let (PatKind::Wild, PatKind::Wild) = (&patterns_left[0].kind, &patterns_right[0].kind) {
                        find_good_method_for_match(
                            cx,
                            arms,
                            path_left,
                            path_right,
                            &paths::RESULT_OK,
                            &paths::RESULT_ERR,
                            "is_ok()",
                            "is_err()",
                        )
                        .or_else(|| {
                            find_good_method_for_match(
                                cx,
                                arms,
                                path_left,
                                path_right,
                                &paths::IPADDR_V4,
                                &paths::IPADDR_V6,
                                "is_ipv4()",
                                "is_ipv6()",
                            )
                        })
                    } else {
                        None
                    }
                },
                (PatKind::TupleStruct(ref path_left, patterns, _), PatKind::Path(ref path_right))
                | (PatKind::Path(ref path_left), PatKind::TupleStruct(ref path_right, patterns, _))
                    if patterns.len() == 1 =>
                {
                    if let PatKind::Wild = patterns[0].kind {
                        find_good_method_for_match(
                            cx,
                            arms,
                            path_left,
                            path_right,
                            &paths::OPTION_SOME,
                            &paths::OPTION_NONE,
                            "is_some()",
                            "is_none()",
                        )
                        .or_else(|| {
                            find_good_method_for_match(
                                cx,
                                arms,
                                path_left,
                                path_right,
                                &paths::POLL_READY,
                                &paths::POLL_PENDING,
                                "is_ready()",
                                "is_pending()",
                            )
                        })
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
        cx: &LateContext<'_>,
        arms: &[Arm<'_>],
        path_left: &QPath<'_>,
        path_right: &QPath<'_>,
        expected_left: &[&str],
        expected_right: &[&str],
        should_be_left: &'a str,
        should_be_right: &'a str,
    ) -> Option<&'a str> {
        let body_node_pair = if is_qpath_def_path(cx, path_left, arms[0].pat.hir_id, expected_left)
            && is_qpath_def_path(cx, path_right, arms[1].pat.hir_id, expected_right)
        {
            (&(*arms[0].body).kind, &(*arms[1].body).kind)
        } else if is_qpath_def_path(cx, path_right, arms[1].pat.hir_id, expected_left)
            && is_qpath_def_path(cx, path_left, arms[0].pat.hir_id, expected_right)
        {
            (&(*arms[1].body).kind, &(*arms[0].body).kind)
        } else {
            return None;
        };

        match body_node_pair {
            (ExprKind::Lit(ref lit_left), ExprKind::Lit(ref lit_right)) => match (&lit_left.node, &lit_right.node) {
                (LitKind::Bool(true), LitKind::Bool(false)) => Some(should_be_left),
                (LitKind::Bool(false), LitKind::Bool(true)) => Some(should_be_right),
                _ => None,
            },
            _ => None,
        }
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

/// Implementation of `MATCH_SAME_ARMS`.
fn lint_match_arms<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'_>) {
    if let ExprKind::Match(_, arms, MatchSource::Normal) = expr.kind {
        let hash = |&(_, arm): &(usize, &Arm<'_>)| -> u64 {
            let mut h = SpanlessHash::new(cx);
            h.hash_expr(arm.body);
            h.finish()
        };

        let eq = |&(lindex, lhs): &(usize, &Arm<'_>), &(rindex, rhs): &(usize, &Arm<'_>)| -> bool {
            let min_index = usize::min(lindex, rindex);
            let max_index = usize::max(lindex, rindex);

            let mut local_map: HirIdMap<HirId> = HirIdMap::default();
            let eq_fallback = |a: &Expr<'_>, b: &Expr<'_>| {
                if_chain! {
                    if let Some(a_id) = path_to_local(a);
                    if let Some(b_id) = path_to_local(b);
                    let entry = match local_map.entry(a_id) {
                        Entry::Vacant(entry) => entry,
                        // check if using the same bindings as before
                        Entry::Occupied(entry) => return *entry.get() == b_id,
                    };
                    // the names technically don't have to match; this makes the lint more conservative
                    if cx.tcx.hir().name(a_id) == cx.tcx.hir().name(b_id);
                    if TyS::same_type(cx.typeck_results().expr_ty(a), cx.typeck_results().expr_ty(b));
                    if pat_contains_local(lhs.pat, a_id);
                    if pat_contains_local(rhs.pat, b_id);
                    then {
                        entry.insert(b_id);
                        true
                    } else {
                        false
                    }
                }
            };
            // Arms with a guard are ignored, those can’t always be merged together
            // This is also the case for arms in-between each there is an arm with a guard
            (min_index..=max_index).all(|index| arms[index].guard.is_none())
                && SpanlessEq::new(cx)
                    .expr_fallback(eq_fallback)
                    .eq_expr(lhs.body, rhs.body)
                // these checks could be removed to allow unused bindings
                && bindings_eq(lhs.pat, local_map.keys().copied().collect())
                && bindings_eq(rhs.pat, local_map.values().copied().collect())
        };

        let indexed_arms: Vec<(usize, &Arm<'_>)> = arms.iter().enumerate().collect();
        for (&(_, i), &(_, j)) in search_same(&indexed_arms, hash, eq) {
            span_lint_and_then(
                cx,
                MATCH_SAME_ARMS,
                j.body.span,
                "this `match` has identical arm bodies",
                |diag| {
                    diag.span_note(i.body.span, "same as this");

                    // Note: this does not use `span_suggestion` on purpose:
                    // there is no clean way
                    // to remove the other arm. Building a span and suggest to replace it to ""
                    // makes an even more confusing error message. Also in order not to make up a
                    // span for the whole pattern, the suggestion is only shown when there is only
                    // one pattern. The user should know about `|` if they are already using it…

                    let lhs = snippet(cx, i.pat.span, "<pat1>");
                    let rhs = snippet(cx, j.pat.span, "<pat2>");

                    if let PatKind::Wild = j.pat.kind {
                        // if the last arm is _, then i could be integrated into _
                        // note that i.pat cannot be _, because that would mean that we're
                        // hiding all the subsequent arms, and rust won't compile
                        diag.span_note(
                            i.body.span,
                            &format!(
                                "`{}` has the same arm body as the `_` wildcard, consider removing it",
                                lhs
                            ),
                        );
                    } else {
                        diag.span_help(i.pat.span, &format!("consider refactoring into `{} | {}`", lhs, rhs,))
                            .help("...or consider changing the match arm bodies");
                    }
                },
            );
        }
    }
}

fn pat_contains_local(pat: &Pat<'_>, id: HirId) -> bool {
    let mut result = false;
    pat.walk_short(|p| {
        result |= matches!(p.kind, PatKind::Binding(_, binding_id, ..) if binding_id == id);
        !result
    });
    result
}

/// Returns true if all the bindings in the `Pat` are in `ids` and vice versa
fn bindings_eq(pat: &Pat<'_>, mut ids: HirIdSet) -> bool {
    let mut result = true;
    pat.each_binding_or_first(&mut |_, id, _, _| result &= ids.remove(&id));
    result && ids.is_empty()
}
