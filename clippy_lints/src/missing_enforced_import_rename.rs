use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::paths::{PathNS, lookup_path_str};
use clippy_utils::source::SpanRangeExt;
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::def_id::DefIdMap;
use rustc_hir::{Item, ItemKind, UseKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::TyCtxt;
use rustc_session::impl_lint_pass;
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
}

impl_lint_pass!(ImportRename => [MISSING_ENFORCED_IMPORT_RENAMES]);

impl LateLintPass<'_> for ImportRename {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        if let ItemKind::Use(path, UseKind::Single(_)) = &item.kind {
            for &res in &path.res {
                if let Res::Def(_, id) = res
                    && let Some(name) = self.renames.get(&id)
                    // Remove semicolon since it is not present for nested imports
                    && let span_without_semi = cx.sess().source_map().span_until_char(item.span, ';')
                    && let Some(snip) = span_without_semi.get_source_text(cx)
                    && let Some(import) = match snip.split_once(" as ") {
                        None => Some(snip.as_str()),
                        Some((import, rename)) => {
                            if rename.trim() == name.as_str() {
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
                        format!("{import} as {name}",),
                        Applicability::MachineApplicable,
                    );
                }
            }
        }
    }
}
