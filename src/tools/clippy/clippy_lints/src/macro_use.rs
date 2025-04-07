use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::source::snippet;
use hir::def::{DefKind, Res};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::{self as hir, AmbigArg};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::impl_lint_pass;
use rustc_span::edition::Edition;
use rustc_span::{Span, sym};
use std::collections::BTreeMap;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `#[macro_use] use...`.
    ///
    /// ### Why is this bad?
    /// Since the Rust 2018 edition you can import
    /// macro's directly, this is considered idiomatic.
    ///
    /// ### Example
    /// ```rust,ignore
    /// #[macro_use]
    /// extern crate some_crate;
    ///
    /// fn main() {
    ///     some_macro!();
    /// }
    /// ```
    ///
    /// Use instead:
    ///
    /// ```rust,ignore
    /// use some_crate::some_macro;
    ///
    /// fn main() {
    ///     some_macro!();
    /// }
    /// ```
    #[clippy::version = "1.44.0"]
    pub MACRO_USE_IMPORTS,
    pedantic,
    "#[macro_use] is no longer needed"
}

/// `MacroRefData` includes the name of the macro.
#[derive(Debug, Clone)]
pub struct MacroRefData {
    name: String,
}

impl MacroRefData {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

#[derive(Default)]
pub struct MacroUseImports {
    /// the actual import path used and the span of the attribute above it. The value is
    /// the location, where the lint should be emitted.
    imports: Vec<(String, Span, hir::HirId)>,
    /// the span of the macro reference, kept to ensure only one reference is used per macro call.
    collected: FxHashSet<Span>,
    mac_refs: Vec<MacroRefData>,
}

impl_lint_pass!(MacroUseImports => [MACRO_USE_IMPORTS]);

impl MacroUseImports {
    fn push_unique_macro(&mut self, cx: &LateContext<'_>, span: Span) {
        let call_site = span.source_callsite();
        let name = snippet(cx, cx.sess().source_map().span_until_char(call_site, '!'), "_");
        if span.source_callee().is_some() && !self.collected.contains(&call_site) {
            let name = if name.contains("::") {
                name.split("::").last().unwrap().to_string()
            } else {
                name.to_string()
            };

            self.mac_refs.push(MacroRefData::new(name));
            self.collected.insert(call_site);
        }
    }

    fn push_unique_macro_pat_ty(&mut self, cx: &LateContext<'_>, span: Span) {
        let call_site = span.source_callsite();
        let name = snippet(cx, cx.sess().source_map().span_until_char(call_site, '!'), "_");
        if span.source_callee().is_some() && !self.collected.contains(&call_site) {
            self.mac_refs.push(MacroRefData::new(name.to_string()));
            self.collected.insert(call_site);
        }
    }
}

impl LateLintPass<'_> for MacroUseImports {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &hir::Item<'_>) {
        if cx.sess().opts.edition >= Edition::Edition2018
            && let hir::ItemKind::Use(path, _kind) = &item.kind
            && let hir_id = item.hir_id()
            && let attrs = cx.tcx.hir_attrs(hir_id)
            && let Some(mac_attr) = attrs.iter().find(|attr| attr.has_name(sym::macro_use))
            && let Some(id) = path.res.iter().find_map(|res| match res {
                Res::Def(DefKind::Mod, id) => Some(id),
                _ => None,
            })
            && !id.is_local()
        {
            for kid in cx.tcx.module_children(id) {
                if let Res::Def(DefKind::Macro(_mac_type), mac_id) = kid.res {
                    let span = mac_attr.span();
                    let def_path = cx.tcx.def_path_str(mac_id);
                    self.imports.push((def_path, span, hir_id));
                }
            }
        } else if item.span.from_expansion() {
            self.push_unique_macro_pat_ty(cx, item.span);
        }
    }
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &hir::Expr<'_>) {
        if expr.span.from_expansion() {
            self.push_unique_macro(cx, expr.span);
        }
    }
    fn check_stmt(&mut self, cx: &LateContext<'_>, stmt: &hir::Stmt<'_>) {
        if stmt.span.from_expansion() {
            self.push_unique_macro(cx, stmt.span);
        }
    }
    fn check_pat(&mut self, cx: &LateContext<'_>, pat: &hir::Pat<'_>) {
        if pat.span.from_expansion() {
            self.push_unique_macro_pat_ty(cx, pat.span);
        }
    }
    fn check_ty(&mut self, cx: &LateContext<'_>, ty: &hir::Ty<'_, AmbigArg>) {
        if ty.span.from_expansion() {
            self.push_unique_macro_pat_ty(cx, ty.span);
        }
    }
    fn check_crate_post(&mut self, cx: &LateContext<'_>) {
        let mut used = BTreeMap::new();
        let mut check_dup = vec![];
        for (import, span, hir_id) in &self.imports {
            let found_idx = self.mac_refs.iter().position(|mac| import.ends_with(&mac.name));

            if let Some(idx) = found_idx {
                self.mac_refs.remove(idx);
                let seg = import.split("::").collect::<Vec<_>>();

                match seg.as_slice() {
                    // an empty path is impossible
                    // a path should always consist of 2 or more segments
                    [] | [_] => return,
                    [root, item] => {
                        if !check_dup.contains(&(*item).to_string()) {
                            used.entry((
                                (*root).to_string(),
                                span,
                                hir_id.local_id,
                                cx.tcx.def_path_hash(hir_id.owner.def_id.into()),
                            ))
                            .or_insert_with(|| (vec![], hir_id))
                            .0
                            .push((*item).to_string());
                            check_dup.push((*item).to_string());
                        }
                    },
                    [root, rest @ ..] => {
                        if rest.iter().all(|item| !check_dup.contains(&(*item).to_string())) {
                            let filtered = rest
                                .iter()
                                .filter_map(|item| {
                                    if check_dup.contains(&(*item).to_string()) {
                                        None
                                    } else {
                                        Some((*item).to_string())
                                    }
                                })
                                .collect::<Vec<_>>();
                            used.entry((
                                (*root).to_string(),
                                span,
                                hir_id.local_id,
                                cx.tcx.def_path_hash(hir_id.owner.def_id.into()),
                            ))
                            .or_insert_with(|| (vec![], hir_id))
                            .0
                            .push(filtered.join("::"));
                            check_dup.extend(filtered);
                        } else {
                            let rest = rest.to_vec();
                            used.entry((
                                (*root).to_string(),
                                span,
                                hir_id.local_id,
                                cx.tcx.def_path_hash(hir_id.owner.def_id.into()),
                            ))
                            .or_insert_with(|| (vec![], hir_id))
                            .0
                            .push(rest.join("::"));
                            check_dup.extend(rest.iter().map(ToString::to_string));
                        }
                    },
                }
            }
        }

        // If mac_refs is not empty we have encountered an import we could not handle
        // such as `std::prelude::v1::foo` or some other macro that expands to an import.
        if self.mac_refs.is_empty() {
            for ((root, span, ..), (path, hir_id)) in used {
                let import = if let [single] = &path[..] {
                    format!("{root}::{single}")
                } else {
                    format!("{root}::{{{}}}", path.join(", "))
                };

                span_lint_hir_and_then(
                    cx,
                    MACRO_USE_IMPORTS,
                    *hir_id,
                    *span,
                    "`macro_use` attributes are no longer needed in the Rust 2018 edition",
                    |diag| {
                        diag.span_suggestion(
                            *span,
                            "remove the attribute and import the macro directly, try",
                            format!("use {import};"),
                            Applicability::MaybeIncorrect,
                        );
                    },
                );
            }
        }
    }
}
