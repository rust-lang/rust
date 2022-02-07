use clippy_utils::{meets_msrv, msrvs};
use rustc_hir::{Expr, ExprKind, Local, MatchSource, Pat};
use rustc_lint::{LateContext, LateLintPass};
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};

mod infalliable_detructuring_match;
mod match_as_ref;
mod match_bool;
mod match_like_matches;
mod match_ref_pats;
mod match_same_arms;
mod match_single_binding;
mod match_wild_enum;
mod match_wild_err_arm;
mod overlapping_arms;
mod redundant_pattern_match;
mod rest_pat_in_fully_bound_struct;
mod single_match;
mod wild_in_or_pats;

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
            overlapping_arms::check(cx, ex, arms);
            match_wild_err_arm::check(cx, ex, arms);
            match_wild_enum::check(cx, ex, arms);
            match_as_ref::check(cx, ex, arms, expr);
            wild_in_or_pats::check(cx, arms);

            if self.infallible_destructuring_match_linted {
                self.infallible_destructuring_match_linted = false;
            } else {
                match_single_binding::check(cx, ex, arms, expr);
            }
        }
        if let ExprKind::Match(ex, arms, _) = expr.kind {
            match_ref_pats::check(cx, ex, arms.iter().map(|el| el.pat), expr);
        }
    }

    fn check_local(&mut self, cx: &LateContext<'tcx>, local: &'tcx Local<'_>) {
        self.infallible_destructuring_match_linted |= infalliable_detructuring_match::check(cx, local);
    }

    fn check_pat(&mut self, cx: &LateContext<'tcx>, pat: &'tcx Pat<'_>) {
        rest_pat_in_fully_bound_struct::check(cx, pat);
    }

    extract_msrv_attr!(LateContext);
}
