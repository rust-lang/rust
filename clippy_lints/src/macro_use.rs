use crate::utils::{in_macro, snippet, span_lint_and_sugg};
use hir::def::{DefKind, Res};
use if_chain::if_chain;
use rustc_ast::ast;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{edition::Edition, Span};

declare_clippy_lint! {
    /// **What it does:** Checks for `#[macro_use] use...`.
    ///
    /// **Why is this bad?** Since the Rust 2018 edition you can import
    /// macro's directly, this is considered idiomatic.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// #[macro_use]
    /// use lazy_static;
    /// ```
    pub MACRO_USE_IMPORTS,
    pedantic,
    "#[macro_use] is no longer needed"
}

const BRACKETS: &[char] = &['<', '>'];

/// `MacroRefData` includes the name of the macro
/// and the path from `SourceMap::span_to_filename`.
#[derive(Debug, Clone)]
pub struct MacroRefData {
    name: String,
    path: String,
}

impl MacroRefData {
    pub fn new(name: &str, callee: Span, cx: &LateContext<'_, '_>) -> Self {
        let mut path = cx.sess().source_map().span_to_filename(callee).to_string();

        // std lib paths are <::std::module::file type>
        // so remove brackets, space and type.
        if path.contains('<') {
            path = path.replace(BRACKETS, "");
        }
        if path.contains(' ') {
            path = path.split(' ').next().unwrap().to_string();
        }
        Self {
            name: name.to_string(),
            path,
        }
    }
}

#[derive(Default)]
#[allow(clippy::module_name_repetitions)]
pub struct MacroUseImports {
    /// the actual import path used and the span of the attribute above it.
    imports: Vec<(String, Span)>,
    /// the span of the macro reference, kept to ensure only one reference is used per macro call.
    collected: FxHashSet<Span>,
    mac_refs: Vec<MacroRefData>,
}

impl_lint_pass!(MacroUseImports => [MACRO_USE_IMPORTS]);

impl MacroUseImports {
    fn push_unique_macro(&mut self, cx: &LateContext<'_, '_>, name: &str, call_site: Span, callee: Span) {
        if !self.collected.contains(&call_site) {
            let name = if name.contains("::") {
                name.split("::").last().unwrap().to_string()
            } else {
                name.to_string()
            };

            self.mac_refs.push(MacroRefData::new(&name, callee, cx));
            self.collected.insert(call_site);
        }
    }

    fn push_unique_macro_pat_ty(&mut self, cx: &LateContext<'_, '_>, name: &str, call_site: Span, callee: Span) {
        if !self.collected.contains(&call_site) {
            self.mac_refs.push(MacroRefData::new(&name, callee, cx));
            self.collected.insert(call_site);
        }
    }
}

impl<'l, 'txc> LateLintPass<'l, 'txc> for MacroUseImports {
    fn check_item(&mut self, cx: &LateContext<'_, '_>, item: &hir::Item<'_>) {
        if_chain! {
                if cx.sess().opts.edition == Edition::Edition2018;
                if let hir::ItemKind::Use(path, _kind) = &item.kind;
                if let Some(mac_attr) = item
                    .attrs
                    .iter()
                    .find(|attr| attr.ident().map(|s| s.to_string()) == Some("macro_use".to_string()));
                if let Res::Def(DefKind::Mod, id) = path.res;
                then {
                    for kid in cx.tcx.item_children(id).iter() {
                        if let Res::Def(DefKind::Macro(_mac_type), mac_id) = kid.res {
                            let span = mac_attr.span;
                            self.imports.push((cx.tcx.def_path_str(mac_id), span));
                        }
                    }
                } else {
                    if in_macro(item.span) {
                        let call_site = item.span.source_callsite();
                        let name = snippet(cx, cx.sess().source_map().span_until_char(call_site, '!'), "_");
                        if let Some(callee) = item.span.source_callee() {
                            if !self.collected.contains(&call_site) {
                                self.mac_refs.push(MacroRefData::new(&name, callee.def_site, cx));
                                self.collected.insert(call_site);
                            }
                        }
                    }
            }
        }
    }
    fn check_attribute(&mut self, cx: &LateContext<'_, '_>, attr: &ast::Attribute) {
        if in_macro(attr.span) {
            let call_site = attr.span.source_callsite();
            let name = snippet(cx, cx.sess().source_map().span_until_char(call_site, '!'), "_");
            if let Some(callee) = attr.span.source_callee() {
                self.push_unique_macro(cx, &name, call_site, callee.def_site);
            }
        }
    }
    fn check_expr(&mut self, cx: &LateContext<'_, '_>, expr: &hir::Expr<'_>) {
        if in_macro(expr.span) {
            let call_site = expr.span.source_callsite();
            let name = snippet(cx, cx.sess().source_map().span_until_char(call_site, '!'), "_");
            if let Some(callee) = expr.span.source_callee() {
                self.push_unique_macro(cx, &name, call_site, callee.def_site);
            }
        }
    }
    fn check_stmt(&mut self, cx: &LateContext<'_, '_>, stmt: &hir::Stmt<'_>) {
        if in_macro(stmt.span) {
            let call_site = stmt.span.source_callsite();
            let name = snippet(cx, cx.sess().source_map().span_until_char(call_site, '!'), "_");
            if let Some(callee) = stmt.span.source_callee() {
                self.push_unique_macro(cx, &name, call_site, callee.def_site);
            }
        }
    }
    fn check_pat(&mut self, cx: &LateContext<'_, '_>, pat: &hir::Pat<'_>) {
        if in_macro(pat.span) {
            let call_site = pat.span.source_callsite();
            let name = snippet(cx, cx.sess().source_map().span_until_char(call_site, '!'), "_");
            if let Some(callee) = pat.span.source_callee() {
                self.push_unique_macro_pat_ty(cx, &name, call_site, callee.def_site);
            }
        }
    }
    fn check_ty(&mut self, cx: &LateContext<'_, '_>, ty: &hir::Ty<'_>) {
        if in_macro(ty.span) {
            let call_site = ty.span.source_callsite();
            let name = snippet(cx, cx.sess().source_map().span_until_char(call_site, '!'), "_");
            if let Some(callee) = ty.span.source_callee() {
                self.push_unique_macro_pat_ty(cx, &name, call_site, callee.def_site);
            }
        }
    }

    fn check_crate_post(&mut self, cx: &LateContext<'_, '_>, _krate: &hir::Crate<'_>) {
        for (import, span) in &self.imports {
            let matched = self.mac_refs.iter().any(|mac| import.ends_with(&mac.name));

            if matched {
                self.mac_refs.retain(|mac| !import.ends_with(&mac.name));
                let msg = "`macro_use` attributes are no longer needed in the Rust 2018 edition";
                let help = format!("use {}", import);
                span_lint_and_sugg(
                    cx,
                    MACRO_USE_IMPORTS,
                    *span,
                    msg,
                    "remove the attribute and import the macro directly, try",
                    help,
                    Applicability::MaybeIncorrect,
                )
            }
        }
        if !self.mac_refs.is_empty() {
            // TODO if not empty we found one we could not make a suggestion for
            // such as std::prelude::v1 or something else I haven't thought of.
            // If we defer the calling of span_lint_and_sugg we can make a decision about its
            // applicability?
        }
    }
}
