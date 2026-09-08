use clippy_utils::diagnostics::span_lint_and_then;
use rustc_ast::{Item, ItemKind, UseTree, UseTreeKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass, declare_lint_pass};
use rustc_span::symbol::kw;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for imports ending in `::{self}`.
    ///
    /// ### Why restrict this?
    /// In most cases, this can be written much more cleanly by omitting `::{self}`.
    ///
    /// ### Known problems
    /// Removing `::{self}` will cause any non-module items at the same path to also be imported.
    /// This might cause a naming conflict (https://github.com/rust-lang/rustfmt/issues/3568). This lint makes no attempt
    /// to detect this scenario and that is why it is a restriction lint.
    ///
    /// ### Example
    /// ```no_run
    /// use std::io::{self};
    /// ```
    /// Use instead:
    /// ```no_run
    /// use std::io;
    /// ```
    #[clippy::version = "1.53.0"]
    pub UNNECESSARY_SELF_IMPORTS,
    restriction,
    "imports ending in `::{self}`, which can be omitted"
}

declare_lint_pass!(UnnecessarySelfImports => [UNNECESSARY_SELF_IMPORTS]);

impl EarlyLintPass for UnnecessarySelfImports {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        let ItemKind::Use(use_tree) = &item.kind else { return };
        for_each_self_import(use_tree, |self_import| {
            let SelfImport {
                tree,
                self_tree,
                is_toplevel,
            } = self_import;
            let Some(last_segment) = tree.prefix.segments.last() else {
                return;
            };

            span_lint_and_then(
                cx,
                UNNECESSARY_SELF_IMPORTS,
                // If this is the top-level import, highlight the entirety of it,
                // i.e. the `use ` and `;` as well
                if is_toplevel { item.span } else { tree.span() },
                "import ending with `::{self}`",
                |diag| {
                    diag.span_suggestion(
                        last_segment.span().to(tree.hi_span()),
                        "consider omitting `::{self}`",
                        format!(
                            "{}{}",
                            last_segment.ident,
                            if let UseTreeKind::Simple(Some(alias)) = self_tree.kind {
                                format!(" as {alias}")
                            } else {
                                String::new()
                            },
                        ),
                        Applicability::MaybeIncorrect,
                    );
                    diag.note("this will slightly change semantics; any non-module items at the same path will also be imported");
                },
            );
        });
    }
}

/// An import ending in `::{self}`
///
/// ```no_run
///    use std::io::{self};
/// //               ^^^^  self_tree
/// // ^^^^^^^^^^^^^^^^^^^ tree
/// ```
struct SelfImport<'a> {
    tree: &'a UseTree,
    self_tree: &'a UseTree,
    /// Whether this is the top-level `use` item:
    /// ```no_run
    /// use std::io::{self};
    /// //       ^^^^^^^^^^
    /// ```
    /// or not:
    /// ```no_run
    /// use std::{
    ///     io::{self},
    /// //  ^^^^^^^^^^
    /// };
    /// ```
    is_toplevel: bool,
}

/// Traverses the `use` tree and calls `emit_lint` for every `self` import found
// XXX: rewrite as a generator returning `SelfImport`s, if those ever get stabilized
fn for_each_self_import<'a>(tree: &'a UseTree, emit_lint: impl Fn(SelfImport<'a>) + Copy) {
    fn inner<'a>(tree: &'a UseTree, emit_lint: impl Fn(SelfImport<'a>) + Copy, is_toplevel: bool) {
        if let UseTreeKind::Nested { items, .. } = &tree.kind {
            if let [(self_tree, _)] = &**items
                && let [self_seg] = &*self_tree.prefix.segments
                && self_seg.ident.name == kw::SelfLower
            {
                emit_lint(SelfImport {
                    tree,
                    self_tree,
                    is_toplevel,
                });
            } else {
                for (subtree, _) in &**items {
                    inner(subtree, emit_lint, false);
                }
            }
        }
    }
    inner(tree, emit_lint, true);
}
