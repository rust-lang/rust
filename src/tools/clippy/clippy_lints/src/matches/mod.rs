mod collapsible_match;
mod infallible_destructuring_match;
mod manual_filter;
mod manual_map;
mod manual_ok_err;
mod manual_unwrap_or;
mod manual_utils;
mod match_as_ref;
mod match_bool;
mod match_like_matches;
mod match_ref_pats;
mod match_same_arms;
mod match_single_binding;
mod match_str_case_mismatch;
mod match_wild_enum;
mod match_wild_err_arm;
mod needless_match;
mod overlapping_arms;
mod redundant_guards;
mod redundant_pattern_match;
mod rest_pat_in_fully_bound_struct;
mod significant_drop_in_scrutinee;
mod single_match;
mod try_err;
mod wild_in_or_pats;

use clippy_config::Conf;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::walk_span_to_context;
use clippy_utils::{
    higher, is_direct_expn_of, is_in_const_context, is_span_match, span_contains_cfg, span_extract_comments,
};
use rustc_hir::{Arm, Expr, ExprKind, LetStmt, MatchSource, Pat, PatKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::impl_lint_pass;
use rustc_span::{SpanData, SyntaxContext};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for matches with a single arm where an `if let`
    /// will usually suffice.
    ///
    /// This intentionally does not lint if there are comments
    /// inside of the other arm, so as to allow the user to document
    /// why having another explicit pattern with an empty body is necessary,
    /// or because the comments need to be preserved for other reasons.
    ///
    /// ### Why is this bad?
    /// Just readability – `if let` nests less than a `match`.
    ///
    /// ### Example
    /// ```no_run
    /// # fn bar(stool: &str) {}
    /// # let x = Some("abc");
    /// match x {
    ///     Some(ref foo) => bar(foo),
    ///     _ => (),
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # fn bar(stool: &str) {}
    /// # let x = Some("abc");
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
    /// ```no_run
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
    /// ```no_run
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
    /// match x {
    ///     &A(ref y) => foo(y),
    ///     &B => bar(),
    ///     _ => frob(&x),
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
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
    /// ```no_run
    /// # fn foo() {}
    /// # fn bar() {}
    /// let condition: bool = true;
    /// match condition {
    ///     true => foo(),
    ///     false => bar(),
    /// }
    /// ```
    /// Use if/else instead:
    /// ```no_run
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
    /// ```no_run
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
    /// ```no_run
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
    /// ```no_run
    /// let x: Option<()> = None;
    ///
    /// let r: Option<&()> = match x {
    ///     None => None,
    ///     Some(ref v) => Some(v),
    /// };
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let x: Option<()> = None;
    ///
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
    /// ### Why restrict this?
    /// New enum variants added by library updates can be missed.
    ///
    /// ### Known problems
    /// Suggested replacements may be incorrect if guards exhaustively cover some
    /// variants, and also may not use correct path to enum if it's not present in the current scope.
    ///
    /// ### Example
    /// ```no_run
    /// # enum Foo { A(usize), B(usize) }
    /// # let x = Foo::B(1);
    /// match x {
    ///     Foo::A(_) => {},
    ///     _ => {},
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # enum Foo { A(usize), B(usize) }
    /// # let x = Foo::B(1);
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
    /// ```no_run
    /// # enum Foo { A, B, C }
    /// # let x = Foo::B;
    /// match x {
    ///     Foo::A => {},
    ///     Foo::B => {},
    ///     _ => {},
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # enum Foo { A, B, C }
    /// # let x = Foo::B;
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
    /// ```no_run
    /// # let s = "foo";
    /// match s {
    ///     "a" => {},
    ///     "bar" | _ => {},
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let s = "foo";
    /// match s {
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
    /// ```no_run
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
    /// ```no_run
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
    /// ```no_run
    /// # let a = 1;
    /// # let b = 2;
    /// match (a, b) {
    ///     (c, d) => {
    ///         // useless match
    ///     }
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let a = 1;
    /// # let b = 2;
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
    /// ### Why restrict this?
    /// Correctness and readability. It's like having a wildcard pattern after
    /// matching all enum variants explicitly.
    ///
    /// ### Example
    /// ```no_run
    /// # struct A { a: i32 }
    /// let a = A { a: 5 };
    ///
    /// match a {
    ///     A { a: 5, .. } => {},
    ///     _ => {},
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # struct A { a: i32 }
    /// # let a = A { a: 5 };
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
    /// `std::task::Poll`, `std::net::IpAddr` or `bool`s
    ///
    /// ### Why is this bad?
    /// It's more concise and clear to just use the proper
    /// utility function or using the condition directly
    ///
    /// ### Known problems
    /// For suggestions involving bindings in patterns, this will change the drop order for the matched type.
    /// Both `if let` and `while let` will drop the value at the end of the block, both `if` and `while` will drop the
    /// value before entering the block. For most types this change will not matter, but for a few
    /// types this will not be an acceptable change (e.g. locks). See the
    /// [reference](https://doc.rust-lang.org/reference/destructors.html#drop-scopes) for more about
    /// drop order.
    ///
    /// ### Example
    /// ```no_run
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
    ///
    /// let cond = true;
    /// if let true = cond {}
    /// matches!(cond, true);
    /// ```
    ///
    /// The more idiomatic use would be:
    ///
    /// ```no_run
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
    ///
    /// let cond = true;
    /// if cond {}
    /// cond;
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
    /// ```no_run
    /// let x = Some(5);
    ///
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
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let x = Some(5);
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
    /// Note: Does not lint on wildcards if the `non_exhaustive_omitted_patterns_lint` feature is
    /// enabled and disallowed.
    ///
    /// ### Why is this bad?
    /// This is probably a copy & paste error. If arm bodies
    /// are the same on purpose, you can factor them
    /// [using `|`](https://doc.rust-lang.org/book/patterns.html#multiple-patterns).
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

declare_clippy_lint! {
    /// ### What it does
    /// Checks for unnecessary `match` or match-like `if let` returns for `Option` and `Result`
    /// when function signatures are the same.
    ///
    /// ### Why is this bad?
    /// This `match` block does nothing and might not be what the coder intended.
    ///
    /// ### Example
    /// ```rust,ignore
    /// fn foo() -> Result<(), i32> {
    ///     match result {
    ///         Ok(val) => Ok(val),
    ///         Err(err) => Err(err),
    ///     }
    /// }
    ///
    /// fn bar() -> Option<i32> {
    ///     if let Some(val) = option {
    ///         Some(val)
    ///     } else {
    ///         None
    ///     }
    /// }
    /// ```
    ///
    /// Could be replaced as
    ///
    /// ```rust,ignore
    /// fn foo() -> Result<(), i32> {
    ///     result
    /// }
    ///
    /// fn bar() -> Option<i32> {
    ///     option
    /// }
    /// ```
    #[clippy::version = "1.61.0"]
    pub NEEDLESS_MATCH,
    complexity,
    "`match` or match-like `if let` that are unnecessary"
}

declare_clippy_lint! {
    /// ### What it does
    /// Finds nested `match` or `if let` expressions where the patterns may be "collapsed" together
    /// without adding any branches.
    ///
    /// Note that this lint is not intended to find _all_ cases where nested match patterns can be merged, but only
    /// cases where merging would most likely make the code more readable.
    ///
    /// ### Why is this bad?
    /// It is unnecessarily verbose and complex.
    ///
    /// ### Example
    /// ```no_run
    /// fn func(opt: Option<Result<u64, String>>) {
    ///     let n = match opt {
    ///         Some(n) => match n {
    ///             Ok(n) => n,
    ///             _ => return,
    ///         }
    ///         None => return,
    ///     };
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn func(opt: Option<Result<u64, String>>) {
    ///     let n = match opt {
    ///         Some(Ok(n)) => n,
    ///         _ => return,
    ///     };
    /// }
    /// ```
    #[clippy::version = "1.50.0"]
    pub COLLAPSIBLE_MATCH,
    style,
    "Nested `match` or `if let` expressions where the patterns may be \"collapsed\" together."
}

declare_clippy_lint! {
    /// ### What it does
    /// Finds patterns that reimplement `Option::unwrap_or` or `Result::unwrap_or`.
    ///
    /// ### Why is this bad?
    /// Concise code helps focusing on behavior instead of boilerplate.
    ///
    /// ### Example
    /// ```no_run
    /// let foo: Option<i32> = None;
    /// match foo {
    ///     Some(v) => v,
    ///     None => 1,
    /// };
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let foo: Option<i32> = None;
    /// foo.unwrap_or(1);
    /// ```
    #[clippy::version = "1.49.0"]
    pub MANUAL_UNWRAP_OR,
    complexity,
    "finds patterns that can be encoded more concisely with `Option::unwrap_or` or `Result::unwrap_or`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks if a `match` or `if let` expression can be simplified using
    /// `.unwrap_or_default()`.
    ///
    /// ### Why is this bad?
    /// It can be done in one call with `.unwrap_or_default()`.
    ///
    /// ### Example
    /// ```no_run
    /// let x: Option<String> = Some(String::new());
    /// let y: String = match x {
    ///     Some(v) => v,
    ///     None => String::new(),
    /// };
    ///
    /// let x: Option<Vec<String>> = Some(Vec::new());
    /// let y: Vec<String> = if let Some(v) = x {
    ///     v
    /// } else {
    ///     Vec::new()
    /// };
    /// ```
    /// Use instead:
    /// ```no_run
    /// let x: Option<String> = Some(String::new());
    /// let y: String = x.unwrap_or_default();
    ///
    /// let x: Option<Vec<String>> = Some(Vec::new());
    /// let y: Vec<String> = x.unwrap_or_default();
    /// ```
    #[clippy::version = "1.79.0"]
    pub MANUAL_UNWRAP_OR_DEFAULT,
    suspicious,
    "check if a `match` or `if let` can be simplified with `unwrap_or_default`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `match` expressions modifying the case of a string with non-compliant arms
    ///
    /// ### Why is this bad?
    /// The arm is unreachable, which is likely a mistake
    ///
    /// ### Example
    /// ```no_run
    /// # let text = "Foo";
    /// match &*text.to_ascii_lowercase() {
    ///     "foo" => {},
    ///     "Bar" => {},
    ///     _ => {},
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// # let text = "Foo";
    /// match &*text.to_ascii_lowercase() {
    ///     "foo" => {},
    ///     "bar" => {},
    ///     _ => {},
    /// }
    /// ```
    #[clippy::version = "1.58.0"]
    pub MATCH_STR_CASE_MISMATCH,
    correctness,
    "creation of a case altering match expression with non-compliant arms"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for temporaries returned from function calls in a match scrutinee that have the
    /// `clippy::has_significant_drop` attribute.
    ///
    /// ### Why is this bad?
    /// The `clippy::has_significant_drop` attribute can be added to types whose Drop impls have
    /// an important side-effect, such as unlocking a mutex, making it important for users to be
    /// able to accurately understand their lifetimes. When a temporary is returned in a function
    /// call in a match scrutinee, its lifetime lasts until the end of the match block, which may
    /// be surprising.
    ///
    /// For `Mutex`es this can lead to a deadlock. This happens when the match scrutinee uses a
    /// function call that returns a `MutexGuard` and then tries to lock again in one of the match
    /// arms. In that case the `MutexGuard` in the scrutinee will not be dropped until the end of
    /// the match block and thus will not unlock.
    ///
    /// ### Example
    /// ```rust,ignore
    /// # use std::sync::Mutex;
    /// # struct State {}
    /// # impl State {
    /// #     fn foo(&self) -> bool {
    /// #         true
    /// #     }
    /// #     fn bar(&self) {}
    /// # }
    /// let mutex = Mutex::new(State {});
    ///
    /// match mutex.lock().unwrap().foo() {
    ///     true => {
    ///         mutex.lock().unwrap().bar(); // Deadlock!
    ///     }
    ///     false => {}
    /// };
    ///
    /// println!("All done!");
    /// ```
    /// Use instead:
    /// ```no_run
    /// # use std::sync::Mutex;
    /// # struct State {}
    /// # impl State {
    /// #     fn foo(&self) -> bool {
    /// #         true
    /// #     }
    /// #     fn bar(&self) {}
    /// # }
    /// let mutex = Mutex::new(State {});
    ///
    /// let is_foo = mutex.lock().unwrap().foo();
    /// match is_foo {
    ///     true => {
    ///         mutex.lock().unwrap().bar();
    ///     }
    ///     false => {}
    /// };
    ///
    /// println!("All done!");
    /// ```
    #[clippy::version = "1.60.0"]
    pub SIGNIFICANT_DROP_IN_SCRUTINEE,
    nursery,
    "warns when a temporary of a type with a drop with a significant side-effect might have a surprising lifetime"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `Err(x)?`.
    ///
    /// ### Why restrict this?
    /// The `?` operator is designed to allow calls that
    /// can fail to be easily chained. For example, `foo()?.bar()` or
    /// `foo(bar()?)`. Because `Err(x)?` can't be used that way (it will
    /// always return), it is more clear to write `return Err(x)`.
    ///
    /// ### Example
    /// ```no_run
    /// fn foo(fail: bool) -> Result<i32, String> {
    ///     if fail {
    ///       Err("failed")?;
    ///     }
    ///     Ok(0)
    /// }
    /// ```
    /// Could be written:
    ///
    /// ```no_run
    /// fn foo(fail: bool) -> Result<i32, String> {
    ///     if fail {
    ///       return Err("failed".into());
    ///     }
    ///     Ok(0)
    /// }
    /// ```
    #[clippy::version = "1.38.0"]
    pub TRY_ERR,
    restriction,
    "return errors explicitly rather than hiding them behind a `?`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `match` which could be implemented using `map`
    ///
    /// ### Why is this bad?
    /// Using the `map` method is clearer and more concise.
    ///
    /// ### Example
    /// ```no_run
    /// match Some(0) {
    ///     Some(x) => Some(x + 1),
    ///     None => None,
    /// };
    /// ```
    /// Use instead:
    /// ```no_run
    /// Some(0).map(|x| x + 1);
    /// ```
    #[clippy::version = "1.52.0"]
    pub MANUAL_MAP,
    style,
    "reimplementation of `map`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `match` which could be implemented using `filter`
    ///
    /// ### Why is this bad?
    /// Using the `filter` method is clearer and more concise.
    ///
    /// ### Example
    /// ```no_run
    /// match Some(0) {
    ///     Some(x) => if x % 2 == 0 {
    ///                     Some(x)
    ///                } else {
    ///                     None
    ///                 },
    ///     None => None,
    /// };
    /// ```
    /// Use instead:
    /// ```no_run
    /// Some(0).filter(|&x| x % 2 == 0);
    /// ```
    #[clippy::version = "1.66.0"]
    pub MANUAL_FILTER,
    complexity,
    "reimplementation of `filter`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for unnecessary guards in match expressions.
    ///
    /// ### Why is this bad?
    /// It's more complex and much less readable. Making it part of the pattern can improve
    /// exhaustiveness checking as well.
    ///
    /// ### Example
    /// ```rust,ignore
    /// match x {
    ///     Some(x) if matches!(x, Some(1)) => ..,
    ///     Some(x) if x == Some(2) => ..,
    ///     _ => todo!(),
    /// }
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// match x {
    ///     Some(Some(1)) => ..,
    ///     Some(Some(2)) => ..,
    ///     _ => todo!(),
    /// }
    /// ```
    #[clippy::version = "1.73.0"]
    pub REDUNDANT_GUARDS,
    complexity,
    "checks for unnecessary guards in match expressions"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for manual implementation of `.ok()` or `.err()`
    /// on `Result` values.
    ///
    /// ### Why is this bad?
    /// Using `.ok()` or `.err()` rather than a `match` or
    /// `if let` is less complex and more readable.
    ///
    /// ### Example
    /// ```no_run
    /// # fn func() -> Result<u32, &'static str> { Ok(0) }
    /// let a = match func() {
    ///     Ok(v) => Some(v),
    ///     Err(_) => None,
    /// };
    /// let b = if let Err(v) = func() {
    ///     Some(v)
    /// } else {
    ///     None
    /// };
    /// ```
    /// Use instead:
    /// ```no_run
    /// # fn func() -> Result<u32, &'static str> { Ok(0) }
    /// let a = func().ok();
    /// let b = func().err();
    /// ```
    #[clippy::version = "1.86.0"]
    pub MANUAL_OK_ERR,
    complexity,
    "find manual implementations of `.ok()` or `.err()` on `Result`"
}

pub struct Matches {
    msrv: Msrv,
    infallible_destructuring_match_linted: bool,
}

impl Matches {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            msrv: conf.msrv,
            infallible_destructuring_match_linted: false,
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
    NEEDLESS_MATCH,
    COLLAPSIBLE_MATCH,
    MANUAL_UNWRAP_OR,
    MANUAL_UNWRAP_OR_DEFAULT,
    MATCH_STR_CASE_MISMATCH,
    SIGNIFICANT_DROP_IN_SCRUTINEE,
    TRY_ERR,
    MANUAL_MAP,
    MANUAL_FILTER,
    REDUNDANT_GUARDS,
    MANUAL_OK_ERR,
]);

impl<'tcx> LateLintPass<'tcx> for Matches {
    #[expect(clippy::too_many_lines)]
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if is_direct_expn_of(expr.span, "matches").is_none() && expr.span.in_external_macro(cx.sess().source_map()) {
            return;
        }
        let from_expansion = expr.span.from_expansion();

        if let ExprKind::Match(ex, arms, source) = expr.kind {
            if is_direct_expn_of(expr.span, "matches").is_some()
                && let [arm, _] = arms
            {
                redundant_pattern_match::check_match(cx, expr, ex, arms);
                redundant_pattern_match::check_matches_true(cx, expr, arm, ex);
            }

            if source == MatchSource::Normal && !is_span_match(cx, expr.span) {
                return;
            }
            if matches!(source, MatchSource::Normal | MatchSource::ForLoopDesugar) {
                significant_drop_in_scrutinee::check_match(cx, expr, ex, arms, source);
            }

            collapsible_match::check_match(cx, arms, self.msrv);
            if !from_expansion {
                // These don't depend on a relationship between multiple arms
                match_wild_err_arm::check(cx, ex, arms);
                wild_in_or_pats::check(cx, ex, arms);
            }

            if let MatchSource::TryDesugar(_) = source {
                try_err::check(cx, expr, ex);
            }

            if !from_expansion && !contains_cfg_arm(cx, expr, ex, arms) {
                if source == MatchSource::Normal {
                    if !(self.msrv.meets(cx, msrvs::MATCHES_MACRO)
                        && match_like_matches::check_match(cx, expr, ex, arms))
                    {
                        match_same_arms::check(cx, arms);
                    }

                    redundant_pattern_match::check_match(cx, expr, ex, arms);
                    let source_map = cx.tcx.sess.source_map();
                    let mut match_comments = span_extract_comments(source_map, expr.span);
                    // We remove comments from inside arms block.
                    if !match_comments.is_empty() {
                        for arm in arms {
                            for comment in span_extract_comments(source_map, arm.body.span) {
                                if let Some(index) = match_comments
                                    .iter()
                                    .enumerate()
                                    .find(|(_, cm)| **cm == comment)
                                    .map(|(index, _)| index)
                                {
                                    match_comments.remove(index);
                                }
                            }
                        }
                    }
                    // If there are still comments, it means they are outside of the arms. Tell the lint
                    // code about it.
                    single_match::check(cx, ex, arms, expr, !match_comments.is_empty());
                    match_bool::check(cx, ex, arms, expr);
                    overlapping_arms::check(cx, ex, arms);
                    match_wild_enum::check(cx, ex, arms);
                    match_as_ref::check(cx, ex, arms, expr);
                    needless_match::check_match(cx, ex, arms, expr);
                    match_str_case_mismatch::check(cx, ex, arms);
                    redundant_guards::check(cx, arms, self.msrv);

                    if !is_in_const_context(cx) {
                        manual_unwrap_or::check_match(cx, expr, ex, arms);
                        manual_map::check_match(cx, expr, ex, arms);
                        manual_filter::check_match(cx, ex, arms, expr);
                        manual_ok_err::check_match(cx, expr, ex, arms);
                    }

                    if self.infallible_destructuring_match_linted {
                        self.infallible_destructuring_match_linted = false;
                    } else {
                        match_single_binding::check(cx, ex, arms, expr);
                    }
                }
                match_ref_pats::check(cx, ex, arms.iter().map(|el| el.pat), expr);
            }
        } else if let Some(if_let) = higher::IfLet::hir(cx, expr) {
            collapsible_match::check_if_let(cx, if_let.let_pat, if_let.if_then, if_let.if_else, self.msrv);
            significant_drop_in_scrutinee::check_if_let(cx, expr, if_let.let_expr, if_let.if_then, if_let.if_else);
            if !from_expansion {
                if let Some(else_expr) = if_let.if_else {
                    if self.msrv.meets(cx, msrvs::MATCHES_MACRO) {
                        match_like_matches::check_if_let(
                            cx,
                            expr,
                            if_let.let_pat,
                            if_let.let_expr,
                            if_let.if_then,
                            else_expr,
                        );
                    }
                    if !is_in_const_context(cx) {
                        manual_unwrap_or::check_if_let(
                            cx,
                            expr,
                            if_let.let_pat,
                            if_let.let_expr,
                            if_let.if_then,
                            else_expr,
                        );
                        manual_map::check_if_let(cx, expr, if_let.let_pat, if_let.let_expr, if_let.if_then, else_expr);
                        manual_filter::check_if_let(
                            cx,
                            expr,
                            if_let.let_pat,
                            if_let.let_expr,
                            if_let.if_then,
                            else_expr,
                        );
                        manual_ok_err::check_if_let(
                            cx,
                            expr,
                            if_let.let_pat,
                            if_let.let_expr,
                            if_let.if_then,
                            else_expr,
                        );
                    }
                }
                redundant_pattern_match::check_if_let(
                    cx,
                    expr,
                    if_let.let_pat,
                    if_let.let_expr,
                    if_let.if_else.is_some(),
                    if_let.let_span,
                );
                needless_match::check_if_let(cx, expr, &if_let);
            }
        } else {
            if let Some(while_let) = higher::WhileLet::hir(expr) {
                significant_drop_in_scrutinee::check_while_let(cx, expr, while_let.let_expr, while_let.if_then);
            }
            if !from_expansion {
                redundant_pattern_match::check(cx, expr);
            }
        }
    }

    fn check_local(&mut self, cx: &LateContext<'tcx>, local: &'tcx LetStmt<'_>) {
        self.infallible_destructuring_match_linted |=
            local.els.is_none() && infallible_destructuring_match::check(cx, local);
    }

    fn check_pat(&mut self, cx: &LateContext<'tcx>, pat: &'tcx Pat<'_>) {
        rest_pat_in_fully_bound_struct::check(cx, pat);
    }
}

/// Checks if there are any arms with a `#[cfg(..)]` attribute.
fn contains_cfg_arm(cx: &LateContext<'_>, e: &Expr<'_>, scrutinee: &Expr<'_>, arms: &[Arm<'_>]) -> bool {
    let Some(scrutinee_span) = walk_span_to_context(scrutinee.span, SyntaxContext::root()) else {
        // Shouldn't happen, but treat this as though a `cfg` attribute were found
        return true;
    };

    let start = scrutinee_span.hi();
    let mut arm_spans = arms.iter().map(|arm| {
        let data = arm.span.data();
        (data.ctxt == SyntaxContext::root()).then_some((data.lo, data.hi))
    });
    let end = e.span.hi();

    // Walk through all the non-code space before each match arm. The space trailing the final arm is
    // handled after the `try_fold` e.g.
    //
    // match foo {
    // _________^-                      everything between the scrutinee and arm1
    //|    arm1 => (),
    //|---^___________^                 everything before arm2
    //|    #[cfg(feature = "enabled")]
    //|    arm2 => some_code(),
    //|---^____________________^        everything before arm3
    //|    // some comment about arm3
    //|    arm3 => some_code(),
    //|---^____________________^        everything after arm3
    //|    #[cfg(feature = "disabled")]
    //|    arm4 = some_code(),
    //|};
    //|^
    let found = arm_spans.try_fold(start, |start, range| {
        let Some((end, next_start)) = range else {
            // Shouldn't happen as macros can't expand to match arms, but treat this as though a `cfg` attribute
            // were found.
            return Err(());
        };
        let span = SpanData {
            lo: start,
            hi: end,
            ctxt: SyntaxContext::root(),
            parent: None,
        }
        .span();
        (!span_contains_cfg(cx, span)).then_some(next_start).ok_or(())
    });
    match found {
        Ok(start) => {
            let span = SpanData {
                lo: start,
                hi: end,
                ctxt: SyntaxContext::root(),
                parent: None,
            }
            .span();
            span_contains_cfg(cx, span)
        },
        Err(()) => true,
    }
}

/// Checks if `pat` contains OR patterns that cannot be nested due to a too low MSRV.
fn pat_contains_disallowed_or(cx: &LateContext<'_>, pat: &Pat<'_>, msrv: Msrv) -> bool {
    let mut contains_or = false;
    pat.walk(|p| {
        let is_or = matches!(p.kind, PatKind::Or(_));
        contains_or |= is_or;
        !is_or
    });
    contains_or && !msrv.meets(cx, msrvs::OR_PATTERNS)
}
