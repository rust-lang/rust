use clippy_utils::diagnostics::span_lint_and_help;
use rustc_ast::ast::{self, Inline, ItemKind, ModKind};
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext as _};
use rustc_session::impl_lint_pass;
use rustc_span::{FileName, SourceFile, sym};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for definitions (structs, functions, traits, etc.) in `mod.rs`
    /// files. `lib.rs` and `main.rs` are not checked.
    ///
    /// ### Why restrict this?
    /// `mod.rs` is well-suited to acting as a table of contents — listing
    /// submodules and re-exports while leaving definitions to named files.
    /// Naming each file after its primary definition keeps filenames
    /// descriptive and unique, makes editor tabs and search results easier
    /// to scan, and makes the filesystem tree mirror the module tree, so
    /// the file layout is uniquely determined by the module structure.
    ///
    /// ### Example
    /// ```ignore
    /// // stuff/mod.rs
    /// mod bar;
    /// pub struct Foo { /* ... */ }
    /// impl Foo { /* ... */ }
    /// ```
    /// Use instead:
    /// ```ignore
    /// // stuff/mod.rs
    /// mod bar;
    /// mod foo;
    /// pub use foo::Foo;
    ///
    /// // stuff/foo.rs
    /// pub struct Foo { /* ... */ }
    /// impl Foo { /* ... */ }
    /// ```
    ///
    /// ### Notes
    /// This lint is most useful alongside `self_named_module_files`, which
    /// requires `mod.rs` files; together they constrain `mod.rs` to
    /// declarations only. Under `mod_module_files` (which forbids `mod.rs`
    /// entirely) this lint has nothing to fire on.
    ///
    /// If a definition's name matches its parent module's name, moving it
    /// produces `foo/foo.rs`, which `module_inception` flags — projects
    /// in that situation typically also `allow(module_inception)`.
    #[clippy::version = "1.99.0"]
    pub DEFINITION_IN_MODULE_ROOT,
    restriction,
    "definitions in `mod.rs` should be in named files"
}

impl_lint_pass!(DefinitionInModuleRoot => [DEFINITION_IN_MODULE_ROOT]);

#[derive(Default)]
pub struct DefinitionInModuleRoot {
    /// Stack tracking whether items at the current nesting level are in a
    /// `mod.rs` file. When the stack is empty, we are at crate root depth
    /// (not linted).
    module_stack: Vec<bool>,
}

impl EarlyLintPass for DefinitionInModuleRoot {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &ast::Item) {
        // Handle module items: push state for their children.
        match &item.kind {
            ItemKind::Mod(.., ModKind::Loaded(_, Inline::No { .. }, mod_spans, ..)) => {
                let file = cx.sess().source_map().lookup_source_file(mod_spans.inner_span.lo());
                self.module_stack.push(is_mod_rs(&file));
                return;
            },
            ItemKind::Mod(..) => {
                // Inline module or unloaded — children are not in a root file.
                self.module_stack.push(false);
                return;
            },
            _ => {},
        }

        // Skip items from macro expansion.
        if item.span.from_expansion() {
            return;
        }

        // Only lint inside mod.rs files (not at crate root depth).
        if !self.module_stack.last().copied().unwrap_or(false) {
            return;
        }

        let Some(kind) = definition_kind(item) else {
            return;
        };

        let help = if let Some(ident) = item.kind.ident() {
            format!("move {kind} `{ident}` to a dedicated file")
        } else {
            format!("move the {kind} to a dedicated file")
        };

        span_lint_and_help(
            cx,
            DEFINITION_IN_MODULE_ROOT,
            item.span,
            "definition in module root file",
            None,
            help,
        );
    }

    fn check_item_post(&mut self, _: &EarlyContext<'_>, item: &ast::Item) {
        if matches!(item.kind, ItemKind::Mod(..)) {
            self.module_stack.pop();
        }
    }
}

/// Returns a human-readable kind string for flagged items, or `None` for
/// items that are allowed in root files (modules, imports, re-exports,
/// `#[macro_export]` macros).
fn definition_kind(item: &ast::Item) -> Option<&'static str> {
    match &item.kind {
        i @ (ItemKind::Struct(..)
        | ItemKind::Enum(..)
        | ItemKind::Union(..)
        | ItemKind::Fn(..)
        | ItemKind::Const(..)
        | ItemKind::Static(..)
        | ItemKind::Impl(..)
        | ItemKind::Trait(..)
        | ItemKind::TraitAlias(..)
        | ItemKind::TyAlias(..)
        | ItemKind::ForeignMod(..)) => Some(i.descr()),
        i @ ItemKind::MacroDef(..) if !has_macro_export(item) => Some(i.descr()),
        ItemKind::ExternCrate(..)
        | ItemKind::Use(..)
        | ItemKind::ConstBlock(..)
        | ItemKind::Mod(..)
        | ItemKind::GlobalAsm(..)
        | ItemKind::MacCall(..)
        | ItemKind::MacroDef(..)
        | ItemKind::Delegation(..)
        | ItemKind::DelegationMac(..) => None,
    }
}

/// Check if an item has `#[macro_export]`.
fn has_macro_export(item: &ast::Item) -> bool {
    item.attrs.iter().any(|attr| attr.has_name(sym::macro_export))
}

/// Check if a source file is `mod.rs`.
fn is_mod_rs(file: &SourceFile) -> bool {
    if let FileName::Real(name) = &file.name {
        name.local_path()
            .and_then(|p| p.file_name())
            .is_some_and(|n| n == "mod.rs")
    } else {
        false
    }
}
