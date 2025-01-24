use clippy_utils::diagnostics::span_lint_and_help;
use rustc_hir::{Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `enum`s with no variants, which therefore are uninhabited types
    /// (cannot be instantiated).
    ///
    /// As of this writing, the `never_type` is still a nightly-only experimental API.
    /// Therefore, this lint is only triggered if `#![feature(never_type)]` is enabled.
    ///
    /// ### Why is this bad?
    /// * If you only want a type which can’t be instantiated, you should use [`!`]
    ///   (the primitive type "never"), because [`!`] has more extensive compiler support
    ///   (type inference, etc.) and implementations of common traits.
    ///
    /// * If you need to introduce a distinct type, consider using a [newtype] `struct`
    ///   containing [`!`] instead (`struct MyType(pub !)`), because it is more idiomatic
    ///   to use a `struct` rather than an `enum` when an `enum` is unnecessary.
    ///
    ///   If you do this, note that the [visibility] of the [`!`] field determines whether
    ///   the uninhabitedness is visible in documentation, and whether it can be pattern
    ///   matched to mark code unreachable. If the field is not visible, then the struct
    ///   acts like any other struct with private fields.
    ///
    /// * If the enum has no variants only because all variants happen to be
    ///   [disabled by conditional compilation][cfg], then it would be appropriate
    ///   to allow the lint, with `#[allow(empty_enum)]`.
    ///
    /// For further information, visit
    /// [the never type’s documentation][`!`].
    ///
    /// ### Example
    /// ```no_run
    /// enum CannotExist {}
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// #![feature(never_type)]
    ///
    /// /// Use the `!` type directly...
    /// type CannotExist = !;
    ///
    /// /// ...or define a newtype which is distinct.
    /// struct CannotExist2(pub !);
    /// ```
    ///
    /// [`!`]: https://doc.rust-lang.org/std/primitive.never.html
    /// [cfg]: https://doc.rust-lang.org/reference/conditional-compilation.html
    /// [newtype]: https://doc.rust-lang.org/book/ch19-04-advanced-types.html#using-the-newtype-pattern-for-type-safety-and-abstraction
    /// [visibility]: https://doc.rust-lang.org/reference/visibility-and-privacy.html
    #[clippy::version = "pre 1.29.0"]
    pub EMPTY_ENUM,
    pedantic,
    "enum with no variants"
}

declare_lint_pass!(EmptyEnum => [EMPTY_ENUM]);

impl LateLintPass<'_> for EmptyEnum {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        if let ItemKind::Enum(..) = item.kind
            // Only suggest the `never_type` if the feature is enabled
            && cx.tcx.features().never_type()
            && let Some(adt) = cx.tcx.type_of(item.owner_id).instantiate_identity().ty_adt_def()
            && adt.variants().is_empty()
        {
            span_lint_and_help(
                cx,
                EMPTY_ENUM,
                item.span,
                "enum with no variants",
                None,
                "consider using the uninhabited type `!` (never type) or a wrapper \
                around it to introduce a type which can't be instantiated",
            );
        }
    }
}
