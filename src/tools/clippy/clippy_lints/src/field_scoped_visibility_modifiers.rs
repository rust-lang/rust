use clippy_utils::diagnostics::span_lint_and_then;
use rustc_ast::ast::{Item, ItemKind, VisibilityKind};
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of scoped visibility modifiers, like `pub(crate)`, on fields. These
    /// make a field visible within a scope between public and private.
    ///
    /// ### Why restrict this?
    /// Scoped visibility modifiers cause a field to be accessible within some scope between
    /// public and private, potentially within an entire crate. This allows for fields to be
    /// non-private while upholding internal invariants, but can be a code smell. Scoped visibility
    /// requires checking a greater area, potentially an entire crate, to verify that an invariant
    /// is upheld, and global analysis requires a lot of effort.
    ///
    /// ### Example
    /// ```no_run
    /// pub mod public_module {
    ///     struct MyStruct {
    ///         pub(crate) first_field: bool,
    ///         pub(super) second_field: bool
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// pub mod public_module {
    ///     struct MyStruct {
    ///         first_field: bool,
    ///         second_field: bool
    ///     }
    ///     impl MyStruct {
    ///         pub(crate) fn get_first_field(&self) -> bool {
    ///             self.first_field
    ///         }
    ///         pub(super) fn get_second_field(&self) -> bool {
    ///             self.second_field
    ///         }
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.81.0"]
    pub FIELD_SCOPED_VISIBILITY_MODIFIERS,
    restriction,
    "checks for usage of a scoped visibility modifier, like `pub(crate)`, on fields"
}

declare_lint_pass!(FieldScopedVisibilityModifiers => [FIELD_SCOPED_VISIBILITY_MODIFIERS]);

impl EarlyLintPass for FieldScopedVisibilityModifiers {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        let ItemKind::Struct(_, _, ref st) = item.kind else {
            return;
        };
        for field in st.fields() {
            let VisibilityKind::Restricted { path, .. } = &field.vis.kind else {
                continue;
            };
            if !path.segments.is_empty() && path.segments[0].ident.name == rustc_span::symbol::kw::SelfLower {
                // pub(self) is equivalent to not using pub at all, so we ignore it
                continue;
            }
            #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
            span_lint_and_then(
                cx,
                FIELD_SCOPED_VISIBILITY_MODIFIERS,
                field.vis.span,
                "scoped visibility modifier on a field",
                |diag| {
                    diag.help("consider making the field private and adding a scoped visibility method for it");
                },
            );
        }
    }
}
