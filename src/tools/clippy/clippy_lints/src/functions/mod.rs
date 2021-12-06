mod must_use;
mod not_unsafe_ptr_arg_deref;
mod result_unit_err;
mod too_many_arguments;
mod too_many_lines;

use rustc_hir as hir;
use rustc_hir::intravisit;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for functions with too many parameters.
    ///
    /// ### Why is this bad?
    /// Functions with lots of parameters are considered bad
    /// style and reduce readability (“what does the 5th parameter mean?”). Consider
    /// grouping some parameters into a new type.
    ///
    /// ### Example
    /// ```rust
    /// # struct Color;
    /// fn foo(x: u32, y: u32, name: &str, c: Color, w: f32, h: f32, a: f32, b: f32) {
    ///     // ..
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub TOO_MANY_ARGUMENTS,
    complexity,
    "functions with too many arguments"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for functions with a large amount of lines.
    ///
    /// ### Why is this bad?
    /// Functions with a lot of lines are harder to understand
    /// due to having to look at a larger amount of code to understand what the
    /// function is doing. Consider splitting the body of the function into
    /// multiple functions.
    ///
    /// ### Example
    /// ```rust
    /// fn im_too_long() {
    ///     println!("");
    ///     // ... 100 more LoC
    ///     println!("");
    /// }
    /// ```
    #[clippy::version = "1.34.0"]
    pub TOO_MANY_LINES,
    pedantic,
    "functions with too many lines"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for public functions that dereference raw pointer
    /// arguments but are not marked `unsafe`.
    ///
    /// ### Why is this bad?
    /// The function should probably be marked `unsafe`, since
    /// for an arbitrary raw pointer, there is no way of telling for sure if it is
    /// valid.
    ///
    /// ### Known problems
    /// * It does not check functions recursively so if the pointer is passed to a
    /// private non-`unsafe` function which does the dereferencing, the lint won't
    /// trigger.
    /// * It only checks for arguments whose type are raw pointers, not raw pointers
    /// got from an argument in some other way (`fn foo(bar: &[*const u8])` or
    /// `some_argument.get_raw_ptr()`).
    ///
    /// ### Example
    /// ```rust,ignore
    /// // Bad
    /// pub fn foo(x: *const u8) {
    ///     println!("{}", unsafe { *x });
    /// }
    ///
    /// // Good
    /// pub unsafe fn foo(x: *const u8) {
    ///     println!("{}", unsafe { *x });
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub NOT_UNSAFE_PTR_ARG_DEREF,
    correctness,
    "public functions dereferencing raw pointer arguments but not marked `unsafe`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for a `#[must_use]` attribute on
    /// unit-returning functions and methods.
    ///
    /// ### Why is this bad?
    /// Unit values are useless. The attribute is likely
    /// a remnant of a refactoring that removed the return type.
    ///
    /// ### Examples
    /// ```rust
    /// #[must_use]
    /// fn useless() { }
    /// ```
    #[clippy::version = "1.40.0"]
    pub MUST_USE_UNIT,
    style,
    "`#[must_use]` attribute on a unit-returning function / method"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for a `#[must_use]` attribute without
    /// further information on functions and methods that return a type already
    /// marked as `#[must_use]`.
    ///
    /// ### Why is this bad?
    /// The attribute isn't needed. Not using the result
    /// will already be reported. Alternatively, one can add some text to the
    /// attribute to improve the lint message.
    ///
    /// ### Examples
    /// ```rust
    /// #[must_use]
    /// fn double_must_use() -> Result<(), ()> {
    ///     unimplemented!();
    /// }
    /// ```
    #[clippy::version = "1.40.0"]
    pub DOUBLE_MUST_USE,
    style,
    "`#[must_use]` attribute on a `#[must_use]`-returning function / method"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for public functions that have no
    /// `#[must_use]` attribute, but return something not already marked
    /// must-use, have no mutable arg and mutate no statics.
    ///
    /// ### Why is this bad?
    /// Not bad at all, this lint just shows places where
    /// you could add the attribute.
    ///
    /// ### Known problems
    /// The lint only checks the arguments for mutable
    /// types without looking if they are actually changed. On the other hand,
    /// it also ignores a broad range of potentially interesting side effects,
    /// because we cannot decide whether the programmer intends the function to
    /// be called for the side effect or the result. Expect many false
    /// positives. At least we don't lint if the result type is unit or already
    /// `#[must_use]`.
    ///
    /// ### Examples
    /// ```rust
    /// // this could be annotated with `#[must_use]`.
    /// fn id<T>(t: T) -> T { t }
    /// ```
    #[clippy::version = "1.40.0"]
    pub MUST_USE_CANDIDATE,
    pedantic,
    "function or method that could take a `#[must_use]` attribute"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for public functions that return a `Result`
    /// with an `Err` type of `()`. It suggests using a custom type that
    /// implements `std::error::Error`.
    ///
    /// ### Why is this bad?
    /// Unit does not implement `Error` and carries no
    /// further information about what went wrong.
    ///
    /// ### Known problems
    /// Of course, this lint assumes that `Result` is used
    /// for a fallible operation (which is after all the intended use). However
    /// code may opt to (mis)use it as a basic two-variant-enum. In that case,
    /// the suggestion is misguided, and the code should use a custom enum
    /// instead.
    ///
    /// ### Examples
    /// ```rust
    /// pub fn read_u8() -> Result<u8, ()> { Err(()) }
    /// ```
    /// should become
    /// ```rust,should_panic
    /// use std::fmt;
    ///
    /// #[derive(Debug)]
    /// pub struct EndOfStream;
    ///
    /// impl fmt::Display for EndOfStream {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         write!(f, "End of Stream")
    ///     }
    /// }
    ///
    /// impl std::error::Error for EndOfStream { }
    ///
    /// pub fn read_u8() -> Result<u8, EndOfStream> { Err(EndOfStream) }
    ///# fn main() {
    ///#     read_u8().unwrap();
    ///# }
    /// ```
    ///
    /// Note that there are crates that simplify creating the error type, e.g.
    /// [`thiserror`](https://docs.rs/thiserror).
    #[clippy::version = "1.49.0"]
    pub RESULT_UNIT_ERR,
    style,
    "public function returning `Result` with an `Err` type of `()`"
}

#[derive(Copy, Clone)]
pub struct Functions {
    too_many_arguments_threshold: u64,
    too_many_lines_threshold: u64,
}

impl Functions {
    pub fn new(too_many_arguments_threshold: u64, too_many_lines_threshold: u64) -> Self {
        Self {
            too_many_arguments_threshold,
            too_many_lines_threshold,
        }
    }
}

impl_lint_pass!(Functions => [
    TOO_MANY_ARGUMENTS,
    TOO_MANY_LINES,
    NOT_UNSAFE_PTR_ARG_DEREF,
    MUST_USE_UNIT,
    DOUBLE_MUST_USE,
    MUST_USE_CANDIDATE,
    RESULT_UNIT_ERR,
]);

impl<'tcx> LateLintPass<'tcx> for Functions {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: intravisit::FnKind<'tcx>,
        decl: &'tcx hir::FnDecl<'_>,
        body: &'tcx hir::Body<'_>,
        span: Span,
        hir_id: hir::HirId,
    ) {
        too_many_arguments::check_fn(cx, kind, decl, span, hir_id, self.too_many_arguments_threshold);
        too_many_lines::check_fn(cx, kind, span, body, self.too_many_lines_threshold);
        not_unsafe_ptr_arg_deref::check_fn(cx, kind, decl, body, hir_id);
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'_>) {
        must_use::check_item(cx, item);
        result_unit_err::check_item(cx, item);
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::ImplItem<'_>) {
        must_use::check_impl_item(cx, item);
        result_unit_err::check_impl_item(cx, item);
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::TraitItem<'_>) {
        too_many_arguments::check_trait_item(cx, item, self.too_many_arguments_threshold);
        not_unsafe_ptr_arg_deref::check_trait_item(cx, item);
        must_use::check_trait_item(cx, item);
        result_unit_err::check_trait_item(cx, item);
    }
}
