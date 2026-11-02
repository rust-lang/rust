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
    /// This might cause a naming conflict ([rust-lang/rustfmt#3568](https://github.com/rust-lang/rustfmt/issues/3568)). This lint makes no attempt
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
                is_nested,
            } = self_import;

            let (sugg_code, sugg_span) = if is_nested && let Some(last_segment) = tree.prefix.segments.last() {
                if let UseTreeKind::Simple(Some(alias)) = self_tree.kind {
                    // Suggest replacing `source::{ident as target}` with `source as target`
                    (
                        format!("{} as {alias}", last_segment.ident),
                        last_segment.span().to(tree.hi_span()),
                    )
                } else {
                    // Suggest removing `::{self}`
                    (String::new(), last_segment.span().shrink_to_hi().to(tree.hi_span()))
                }
            } else if !is_nested && let [.., last_segment, self_segment] = &*self_tree.prefix.segments {
                // Suggest removing `::self`
                (
                    String::new(),
                    last_segment.span().shrink_to_hi().to(self_segment.span()),
                )
            } else {
                return;
            };

            span_lint_and_then(
                cx,
                UNNECESSARY_SELF_IMPORTS,
                // If this is the top-level import, highlight the entirety of it,
                // i.e. the `use ` and `;` as well
                if is_toplevel { item.span } else { tree.span() },
                format!(
                    "import ending with {}",
                    if is_nested { "`::{self}`" } else { "`::self`" }
                ),
                |diag| {
                    diag.span_suggestion(
                        sugg_span,
                        format!(
                            "consider omitting {}",
                            if is_nested { "`::{self}`" } else { "`::self`" }
                        ),
                        sugg_code,
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
    is_nested: bool,
}

/// Traverses the `use` tree and calls `emit_lint` for every `self` import found
// XXX: rewrite as a generator returning `SelfImport`s, if those ever get stabilized
fn for_each_self_import<'a>(tree: &'a UseTree, emit_lint: impl Fn(SelfImport<'a>) + Copy) {
    fn inner<'a>(tree: &'a UseTree, emit_lint: impl Fn(SelfImport<'a>) + Copy, is_toplevel: bool) {
        if let UseTreeKind::Nested { items, .. } = &tree.kind {
            if let [self_tree] = &**items
                && let [self_seg] = &*self_tree.inner.prefix.segments
                && self_seg.ident.name == kw::SelfLower
            {
                emit_lint(SelfImport {
                    tree,
                    self_tree: &self_tree.inner,
                    is_toplevel,
                    is_nested: true,
                });
            } else {
                for subtree in &**items {
                    inner(&subtree.inner, emit_lint, false);
                }
            }
        } else if let UseTreeKind::Simple(_) = &tree.kind
            && tree.prefix.segments.len() > 1
            && let Some(self_seg) = tree.prefix.segments.last()
            && self_seg.ident.name == kw::SelfLower
        {
            emit_lint(SelfImport {
                tree,
                self_tree: tree,
                is_toplevel,
                is_nested: false,
            });
        }
    }
    inner(tree, emit_lint, true);
}
