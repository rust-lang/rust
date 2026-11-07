use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::paths::{PathNS, lookup_path_str};
use clippy_utils::source::SpanExt as _;
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::def_id::DefIdMap;
use rustc_hir::{Item, ItemKind, UseKind, UseTree};
use rustc_lint::{LateContext, LateLintPass, LintContext as _, impl_lint_pass};
use rustc_middle::ty::TyCtxt;
use rustc_span::Symbol;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for imports that do not rename the item as specified
    /// in the `enforced-import-renames` config option.
    ///
    /// Note: Even though this lint is warn-by-default, it will only trigger if
    /// import renames are defined in the `clippy.toml` file.
    ///
    /// ### Why is this bad?
    /// Consistency is important; if a project has defined import renames, then they should be
    /// followed. More practically, some item names are too vague outside of their defining scope,
    /// in which case this can enforce a more meaningful naming.
    ///
    /// ### Example
    /// An example clippy.toml configuration:
    /// ```toml
    /// # clippy.toml
    /// enforced-import-renames = [
    ///     { path = "serde_json::Value", rename = "JsonValue" },
    /// ]
    /// ```
    ///
    /// ```rust,ignore
    /// use serde_json::Value;
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// use serde_json::Value as JsonValue;
    /// ```
    #[clippy::version = "1.55.0"]
    pub MISSING_ENFORCED_IMPORT_RENAMES,
    style,
    "enforce import renames"
}

impl_lint_pass!(ImportRename => [MISSING_ENFORCED_IMPORT_RENAMES]);

pub struct ImportRename {
    renames: DefIdMap<Symbol>,
}

impl ImportRename {
    pub fn new(tcx: TyCtxt<'_>, conf: &'static Conf) -> Self {
        Self {
            renames: conf
                .enforced_import_renames
                .iter()
                .map(|x| (&x.path, Symbol::intern(&x.rename)))
                .flat_map(|(path, rename)| {
                    lookup_path_str(tcx, PathNS::Arbitrary, path)
                        .into_iter()
                        .map(move |id| (id, rename))
                })
                .collect(),
        }
    }

    fn check_use_tree(&mut self, cx: &LateContext<'_>, tree: &UseTree<'_>) {
        let hi = match tree.kind {
            UseKind::Single(ident) => ident.span.hi(),
            UseKind::Glob => return,
            UseKind::Nested { items } => {
                for (tree, ..) in items {
                    self.check_use_tree(cx, tree);
                }
                return;
            },
        };
        // use `present_items` because it could be in any of type_ns, value_ns, macro_ns
        for res in tree.prefix.res.present_items() {
            if let Res::Def(_, id) = res
                && let Some(name) = self.renames.get(&id)
                // Remove semicolon since it is not present for nested imports
                && let span_without_semi = cx.sess().source_map().span_until_char(tree.prefix.span.with_hi(hi), ';')
                && let Some(snip) = span_without_semi.get_text(cx)
                && let Some(import) = match snip.split_once(" as ") {
                    None => Some(snip.as_str()),
                    Some((import, rename)) => {
                        let trimmed_rename = rename.trim();
                        if trimmed_rename == "_" || trimmed_rename == name.as_str() {
                            None
                        } else {
                            Some(import.trim())
                        }
                    },
                }
            {
                span_lint_and_sugg(
                    cx,
                    MISSING_ENFORCED_IMPORT_RENAMES,
                    span_without_semi,
                    "this import should be renamed",
                    "try",
                    format!("{import} as {name}"),
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}

impl LateLintPass<'_> for ImportRename {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        if let ItemKind::Use(tree) = &item.kind {
            self.check_use_tree(cx, tree)
        }
    }
}
