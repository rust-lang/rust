use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::is_from_proc_macro;
use clippy_utils::ty::{implements_trait, is_must_use_ty, match_type};
use clippy_utils::{is_must_use_func_call, paths};
use rustc_hir::{Local, PatKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::GenericArgKind;
use rustc_middle::ty::IsSuggestable;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{BytePos, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `let _ = <expr>` where expr is `#[must_use]`
    ///
    /// ### Why is this bad?
    /// It's better to explicitly handle the value of a `#[must_use]`
    /// expr
    ///
    /// ### Example
    /// ```rust
    /// fn f() -> Result<u32, u32> {
    ///     Ok(0)
    /// }
    ///
    /// let _ = f();
    /// // is_ok() is marked #[must_use]
    /// let _ = f().is_ok();
    /// ```
    #[clippy::version = "1.42.0"]
    pub LET_UNDERSCORE_MUST_USE,
    restriction,
    "non-binding `let` on a `#[must_use]` expression"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `let _ = sync_lock`. This supports `mutex` and `rwlock` in
    /// `parking_lot`. For `std` locks see the `rustc` lint
    /// [`let_underscore_lock`](https://doc.rust-lang.org/nightly/rustc/lints/listing/deny-by-default.html#let-underscore-lock)
    ///
    /// ### Why is this bad?
    /// This statement immediately drops the lock instead of
    /// extending its lifetime to the end of the scope, which is often not intended.
    /// To extend lock lifetime to the end of the scope, use an underscore-prefixed
    /// name instead (i.e. _lock). If you want to explicitly drop the lock,
    /// `std::mem::drop` conveys your intention better and is less error-prone.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let _ = mutex.lock();
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// let _lock = mutex.lock();
    /// ```
    #[clippy::version = "1.43.0"]
    pub LET_UNDERSCORE_LOCK,
    correctness,
    "non-binding `let` on a synchronization lock"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `let _ = <expr>` where the resulting type of expr implements `Future`
    ///
    /// ### Why is this bad?
    /// Futures must be polled for work to be done. The original intention was most likely to await the future
    /// and ignore the resulting value.
    ///
    /// ### Example
    /// ```rust
    /// async fn foo() -> Result<(), ()> {
    ///     Ok(())
    /// }
    /// let _ = foo();
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # async fn context() {
    /// async fn foo() -> Result<(), ()> {
    ///     Ok(())
    /// }
    /// let _ = foo().await;
    /// # }
    /// ```
    #[clippy::version = "1.67.0"]
    pub LET_UNDERSCORE_FUTURE,
    suspicious,
    "non-binding `let` on a future"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `let _ = <expr>` without a type annotation, and suggests to either provide one,
    /// or remove the `let` keyword altogether.
    ///
    /// ### Why is this bad?
    /// The `let _ = <expr>` expression ignores the value of `<expr>` but will remain doing so even
    /// if the type were to change, thus potentially introducing subtle bugs. By supplying a type
    /// annotation, one will be forced to re-visit the decision to ignore the value in such cases.
    ///
    /// ### Known problems
    /// The `_ = <expr>` is not properly supported by some tools (e.g. IntelliJ) and may seem odd
    /// to many developers. This lint also partially overlaps with the other `let_underscore_*`
    /// lints.
    ///
    /// ### Example
    /// ```rust
    /// fn foo() -> Result<u32, ()> {
    ///     Ok(123)
    /// }
    /// let _ = foo();
    /// ```
    /// Use instead:
    /// ```rust
    /// fn foo() -> Result<u32, ()> {
    ///     Ok(123)
    /// }
    /// // Either provide a type annotation:
    /// let _: Result<u32, ()> = foo();
    /// // …or drop the let keyword:
    /// _ = foo();
    /// ```
    #[clippy::version = "1.69.0"]
    pub LET_UNDERSCORE_UNTYPED,
    restriction,
    "non-binding `let` without a type annotation"
}

declare_lint_pass!(LetUnderscore => [LET_UNDERSCORE_MUST_USE, LET_UNDERSCORE_LOCK, LET_UNDERSCORE_FUTURE, LET_UNDERSCORE_UNTYPED]);

const SYNC_GUARD_PATHS: [&[&str]; 3] = [
    &paths::PARKING_LOT_MUTEX_GUARD,
    &paths::PARKING_LOT_RWLOCK_READ_GUARD,
    &paths::PARKING_LOT_RWLOCK_WRITE_GUARD,
];

impl<'tcx> LateLintPass<'tcx> for LetUnderscore {
    fn check_local(&mut self, cx: &LateContext<'tcx>, local: &Local<'tcx>) {
        if !in_external_macro(cx.tcx.sess, local.span)
            && let PatKind::Wild = local.pat.kind
            && let Some(init) = local.init
        {
            let init_ty = cx.typeck_results().expr_ty(init);
            let contains_sync_guard = init_ty.walk().any(|inner| match inner.unpack() {
                GenericArgKind::Type(inner_ty) => SYNC_GUARD_PATHS.iter().any(|path| match_type(cx, inner_ty, path)),
                GenericArgKind::Lifetime(_) | GenericArgKind::Const(_) => false,
            });
            if contains_sync_guard {
                span_lint_and_help(
                    cx,
                    LET_UNDERSCORE_LOCK,
                    local.span,
                    "non-binding `let` on a synchronization lock",
                    None,
                    "consider using an underscore-prefixed named \
                            binding or dropping explicitly with `std::mem::drop`",
                );
            } else if let Some(future_trait_def_id) = cx.tcx.lang_items().future_trait()
                && implements_trait(cx, cx.typeck_results().expr_ty(init), future_trait_def_id, &[]) {
                span_lint_and_help(
                    cx,
                    LET_UNDERSCORE_FUTURE,
                    local.span,
                    "non-binding `let` on a future",
                    None,
                    "consider awaiting the future or dropping explicitly with `std::mem::drop`"
                );
            } else if is_must_use_ty(cx, cx.typeck_results().expr_ty(init)) {
                span_lint_and_help(
                    cx,
                    LET_UNDERSCORE_MUST_USE,
                    local.span,
                    "non-binding `let` on an expression with `#[must_use]` type",
                    None,
                    "consider explicitly using expression value",
                );
            } else if is_must_use_func_call(cx, init) {
                span_lint_and_help(
                    cx,
                    LET_UNDERSCORE_MUST_USE,
                    local.span,
                    "non-binding `let` on a result of a `#[must_use]` function",
                    None,
                    "consider explicitly using function result",
                );
            }

            if local.pat.default_binding_modes && local.ty.is_none() {
                // When `default_binding_modes` is true, the `let` keyword is present.

                // Ignore unnameable types
                if let Some(init) = local.init
                    && !cx.typeck_results().expr_ty(init).is_suggestable(cx.tcx, true)
                {
                    return;
                }

                // Ignore if it is from a procedural macro...
                if is_from_proc_macro(cx, init) {
                    return;
                }

				span_lint_and_help(
                    cx,
                    LET_UNDERSCORE_UNTYPED,
                    local.span,
                    "non-binding `let` without a type annotation",
                    Some(
						Span::new(local.pat.span.hi(),
						local.pat.span.hi() + BytePos(1),
						local.pat.span.ctxt(),
						local.pat.span.parent()
					)),
                    "consider adding a type annotation",
                );
            }
        }
    }
}
