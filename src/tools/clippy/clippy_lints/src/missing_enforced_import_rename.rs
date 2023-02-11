use clippy_utils::{diagnostics::span_lint_and_sugg, source::snippet_opt};

use rustc_data_structures::fx::FxHashMap;
use rustc_errors::Applicability;
use rustc_hir::{def::Res, def_id::DefId, Item, ItemKind, UseKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::Symbol;

use crate::utils::conf::Rename;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for imports that do not rename the item as specified
    /// in the `enforce-import-renames` config option.
    ///
    /// ### Why is this bad?
    /// Consistency is important, if a project has defined import
    /// renames they should be followed. More practically, some item names are too
    /// vague outside of their defining scope this can enforce a more meaningful naming.
    ///
    /// ### Example
    /// An example clippy.toml configuration:
    /// ```toml
    /// # clippy.toml
    /// enforced-import-renames = [ { path = "serde_json::Value", rename = "JsonValue" }]
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
    restriction,
    "enforce import renames"
}

pub struct ImportRename {
    conf_renames: Vec<Rename>,
    renames: FxHashMap<DefId, Symbol>,
}

impl ImportRename {
    pub fn new(conf_renames: Vec<Rename>) -> Self {
        Self {
            conf_renames,
            renames: FxHashMap::default(),
        }
    }
}

impl_lint_pass!(ImportRename => [MISSING_ENFORCED_IMPORT_RENAMES]);

impl LateLintPass<'_> for ImportRename {
    fn check_crate(&mut self, cx: &LateContext<'_>) {
        for Rename { path, rename } in &self.conf_renames {
            let segs = path.split("::").collect::<Vec<_>>();
            for id in clippy_utils::def_path_def_ids(cx, &segs) {
                self.renames.insert(id, Symbol::intern(rename));
            }
        }
    }

    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        if let ItemKind::Use(path, UseKind::Single) = &item.kind {
            for &res in &path.res {
                if_chain! {
                    if let Res::Def(_, id) = res;
                    if let Some(name) = self.renames.get(&id);
                    // Remove semicolon since it is not present for nested imports
                    let span_without_semi = cx.sess().source_map().span_until_char(item.span, ';');
                    if let Some(snip) = snippet_opt(cx, span_without_semi);
                    if let Some(import) = match snip.split_once(" as ") {
                        None => Some(snip.as_str()),
                        Some((import, rename)) => {
                            if rename.trim() == name.as_str() {
                                None
                            } else {
                                Some(import.trim())
                            }
                        },
                    };
                    then {
                        span_lint_and_sugg(
                            cx,
                            MISSING_ENFORCED_IMPORT_RENAMES,
                            span_without_semi,
                            "this import should be renamed",
                            "try",
                            format!(
                                "{import} as {name}",
                            ),
                            Applicability::MachineApplicable,
                        );
                    }
                }
            }
        }
    }
}
