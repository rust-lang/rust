use clippy_utils::diagnostics::span_lint_and_then;
use rustc_ast::ast::{Item, ItemKind};
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks whether some but not all fields of a `struct` are public.
    ///
    /// Either make all fields of a type public, or make none of them public
    ///
    /// ### Why restrict this?
    /// Most types should either be:
    /// * Abstract data types: complex objects with opaque implementation which guard
    ///   interior invariants and expose intentionally limited API to the outside world.
    /// * Data: relatively simple objects which group a bunch of related attributes together,
    ///   but have no invariants.
    ///
    /// ### Example
    /// ```no_run
    /// pub struct Color {
    ///     pub r: u8,
    ///     pub g: u8,
    ///     b: u8,
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// pub struct Color {
    ///     pub r: u8,
    ///     pub g: u8,
    ///     pub b: u8,
    /// }
    /// ```
    #[clippy::version = "1.66.0"]
    pub PARTIAL_PUB_FIELDS,
    restriction,
    "partial fields of a struct are public"
}
declare_lint_pass!(PartialPubFields => [PARTIAL_PUB_FIELDS]);

impl EarlyLintPass for PartialPubFields {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        let ItemKind::Struct(_, _, ref st) = item.kind else {
            return;
        };

        let mut fields = st.fields().iter();
        let Some(first_field) = fields.next() else {
            // Empty struct.
            return;
        };
        let all_pub = first_field.vis.kind.is_pub();
        let all_priv = !all_pub;

        let msg = "mixed usage of pub and non-pub fields";

        for field in fields {
            if all_priv && field.vis.kind.is_pub() {
                #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
                span_lint_and_then(cx, PARTIAL_PUB_FIELDS, field.vis.span, msg, |diag| {
                    diag.help("consider using private field here");
                });
                return;
            } else if all_pub && !field.vis.kind.is_pub() {
                #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
                span_lint_and_then(cx, PARTIAL_PUB_FIELDS, field.vis.span, msg, |diag| {
                    diag.help("consider using public field here");
                });
                return;
            }
        }
    }
}
