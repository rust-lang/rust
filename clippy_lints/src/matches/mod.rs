use clippy_utils::consts::{constant, constant_full_int, miri_to_const, FullInt};
use clippy_utils::diagnostics::{
    multispan_sugg, span_lint_and_help, span_lint_and_note, span_lint_and_sugg, span_lint_and_then,
};
use clippy_utils::macros::{is_panic, root_macro_call};
use clippy_utils::peel_blocks_with_stmt;
use clippy_utils::source::{indent_of, snippet, snippet_block, snippet_opt, snippet_with_applicability};
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::visitors::is_local_used;
use clippy_utils::{
    get_parent_expr, is_lang_ctor, is_refutable, is_wild, meets_msrv, msrvs, path_to_local_id, peel_blocks,
    peel_hir_pat_refs, recurse_or_patterns, strip_pat_refs,
};
use core::iter::once;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::LangItem::{OptionNone, OptionSome};
use rustc_hir::{
    self as hir, Arm, BindingAnnotation, BorrowKind, Expr, ExprKind, Local, MatchSource, Mutability, Node, Pat,
    PatKind, PathSegment, QPath, RangeEnd, TyKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, Ty, VariantDef};
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{sym, symbol::kw, Span};
use std::cmp::Ordering;

mod match_bool;
mod match_like_matches;
mod match_same_arms;
mod redundant_pattern_match;
mod single_match;

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
    #[clippy::version = "pre 1.29.0"]
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
    #[clippy::version = "pre 1.29.0"]
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
    #[clippy::version = "pre 1.29.0"]
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
    #[clippy::version = "pre 1.29.0"]
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
    #[clippy::version = "pre 1.29.0"]
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
    #[clippy::version = "pre 1.29.0"]
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
    #[clippy::version = "pre 1.29.0"]
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
    #[clippy::version = "1.34.0"]
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
    #[clippy::version = "1.45.0"]
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
    #[clippy::version = "1.42.0"]
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
    #[clippy::version = "pre 1.29.0"]
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
    #[clippy::version = "1.43.0"]
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
    #[clippy::version = "1.43.0"]
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
    #[clippy::version = "1.31.0"]
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
    #[clippy::version = "1.47.0"]
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
    #[clippy::version = "pre 1.29.0"]
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
        if expr.span.from_expansion() {
            return;
        }

        redundant_pattern_match::check(cx, expr);

        if meets_msrv(self.msrv.as_ref(), &msrvs::MATCHES_MACRO) {
            if !match_like_matches::check(cx, expr) {
                match_same_arms::check(cx, expr);
            }
        } else {
            match_same_arms::check(cx, expr);
        }

        if let ExprKind::Match(ex, arms, MatchSource::Normal) = expr.kind {
            single_match::check(cx, ex, arms, expr);
            match_bool::check(cx, ex, arms, expr);
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
    }

    fn check_local(&mut self, cx: &LateContext<'tcx>, local: &'tcx Local<'_>) {
        if_chain! {
            if !local.span.from_expansion();
            if let Some(expr) = local.init;
            if let ExprKind::Match(target, arms, MatchSource::Normal) = expr.kind;
            if arms.len() == 1 && arms[0].guard.is_none();
            if let PatKind::TupleStruct(
                QPath::Resolved(None, variant_name), args, _) = arms[0].pat.kind;
            if args.len() == 1;
            if let PatKind::Binding(_, arg, ..) = strip_pat_refs(&args[0]).kind;
            let body = peel_blocks(arms[0].body);
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
            if !pat.span.from_expansion();
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

fn check_overlapping_arms<'tcx>(cx: &LateContext<'tcx>, ex: &'tcx Expr<'_>, arms: &'tcx [Arm<'_>]) {
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

fn check_wild_err_arm<'tcx>(cx: &LateContext<'tcx>, ex: &Expr<'tcx>, arms: &[Arm<'tcx>]) {
    let ex_ty = cx.typeck_results().expr_ty(ex).peel_refs();
    if is_type_diagnostic_item(cx, ex_ty, sym::Result) {
        for arm in arms {
            if let PatKind::TupleStruct(ref path, inner, _) = arm.pat.kind {
                let path_str = rustc_hir_pretty::to_string(rustc_hir_pretty::NO_ANN, |s| s.print_qpath(path, false));
                if path_str == "Err" {
                    let mut matching_wild = inner.iter().any(is_wild);
                    let mut ident_bind_name = kw::Underscore;
                    if !matching_wild {
                        // Looking for unused bindings (i.e.: `_e`)
                        for pat in inner.iter() {
                            if let PatKind::Binding(_, id, ident, None) = pat.kind {
                                if ident.as_str().starts_with('_') && !is_local_used(cx, arm.body, id) {
                                    ident_bind_name = ident.name;
                                    matching_wild = true;
                                }
                            }
                        }
                    }
                    if_chain! {
                        if matching_wild;
                        if let Some(macro_call) = root_macro_call(peel_blocks_with_stmt(arm.body).span);
                        if is_panic(cx, macro_call.def_id);
                        then {
                            // `Err(_)` or `Err(_e)` arm with `panic!` found
                            span_lint_and_note(cx,
                                MATCH_WILD_ERR_ARM,
                                arm.pat.span,
                                &format!("`Err({})` matches all errors", ident_bind_name),
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
impl<'a> CommonPrefixSearcher<'a> {
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
                        Res::Def(
                            DefKind::Const | DefKind::ConstParam | DefKind::AnonConst | DefKind::InlineConst,
                            _,
                        ) => return,
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
                    s.push_str(seg.ident.as_str());
                    s.push_str("::");
                }
                s
            } else {
                let mut s = cx.tcx.def_path_str(adt_def.did);
                s.push_str("::");
                s
            },
            variant.name,
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

fn check_match_ref_pats<'a, 'b, I>(cx: &LateContext<'_>, ex: &Expr<'_>, pats: I, expr: &Expr<'_>)
where
    'b: 'a,
    I: Clone + Iterator<Item = &'a Pat<'b>>,
{
    if !has_multiple_ref_pats(pats.clone()) {
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

#[allow(clippy::too_many_lines)]
fn check_match_single_binding<'a>(cx: &LateContext<'a>, ex: &Expr<'a>, arms: &[Arm<'_>], expr: &Expr<'_>) {
    if expr.span.from_expansion() || arms.len() != 1 || is_refutable(cx, arms[0].pat) {
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
    let match_body = peel_blocks(arms[0].body);
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

/// Gets the ranges for each range pattern arm. Applies `ty` bounds for open ranges.
fn all_ranges<'tcx>(cx: &LateContext<'tcx>, arms: &'tcx [Arm<'_>], ty: Ty<'tcx>) -> Vec<SpannedRange<FullInt>> {
    arms.iter()
        .filter_map(|arm| {
            if let Arm { pat, guard: None, .. } = *arm {
                if let PatKind::Range(ref lhs, ref rhs, range_end) = pat.kind {
                    let lhs_const = match lhs {
                        Some(lhs) => constant(cx, cx.typeck_results(), lhs)?.0,
                        None => miri_to_const(ty.numeric_min_val(cx.tcx)?)?,
                    };
                    let rhs_const = match rhs {
                        Some(rhs) => constant(cx, cx.typeck_results(), rhs)?.0,
                        None => miri_to_const(ty.numeric_max_val(cx.tcx)?)?,
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
        if let ExprKind::Call(e, args) = peel_blocks(arm.body).kind;
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

fn has_multiple_ref_pats<'a, 'b, I>(pats: I) -> bool
where
    'b: 'a,
    I: Iterator<Item = &'a Pat<'b>>,
{
    let mut ref_count = 0;
    for opt in pats.map(|pat| match pat.kind {
        PatKind::Ref(..) => Some(true), // &-patterns
        PatKind::Wild => Some(false),   // an "anything" wildcard is also fine
        _ => None,                      // any other pattern is not fine
    }) {
        if let Some(inner) = opt {
            if inner {
                ref_count += 1;
            }
        } else {
            return false;
        }
    }
    ref_count > 1
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
