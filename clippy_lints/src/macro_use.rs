use crate::utils::{snippet, span_lint_and_sugg, in_macro};
use if_chain::if_chain;
use rustc_ast::ast;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{impl_lint_pass, declare_tool_lint};
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
    pub MACRO_USE_IMPORT,
    pedantic,
    "#[macro_use] is no longer needed"
}

#[derive(Default)]
pub struct MacroUseImport {
    collected: FxHashSet<Span>,
}

impl_lint_pass!(MacroUseImport => [MACRO_USE_IMPORT]);

impl EarlyLintPass for MacroUseImport {

    fn check_item(&mut self, ecx: &EarlyContext<'_>, item: &ast::Item) {
        if_chain! {
            if ecx.sess.opts.edition == Edition::Edition2018;
            if let ast::ItemKind::Use(use_tree) = &item.kind;
            if let Some(mac_attr) = item
                .attrs
                .iter()
                .find(|attr| attr.ident().map(|s| s.to_string()) == Some("macro_use".to_string()));
            then {
                let import_path = snippet(ecx, use_tree.span, "_");
                let mac_names = find_used_macros(ecx, &import_path);
                let msg = "`macro_use` attributes are no longer needed in the Rust 2018 edition";
                let help = format!("use {}::<macro name>", import_path);
                span_lint_and_sugg(
                    ecx,
                    MACRO_USE_IMPORT,
                    mac_attr.span,
                    msg,
                    // "remove the attribute and import the macro directly, try",
                    "",
                    help,
                    Applicability::HasPlaceholders,
                );
            }
        }
    }

    fn check_expr(&mut self, ecx: &EarlyContext<'_>, expr: &ast::Expr) {
        if in_macro(expr.span) {
            let name = snippet(ecx, ecx.sess.source_map().span_until_char(expr.span.source_callsite(), '!'), "_");
            if let Some(callee) = expr.span.source_callee() {
                if self.collected.insert(callee.def_site) {
                    println!("EXPR {:#?}", name);
                }
            }
        }
    }
    fn check_stmt(&mut self, ecx: &EarlyContext<'_>, stmt: &ast::Stmt) {
        if in_macro(stmt.span) {
            let name = snippet(ecx, ecx.sess.source_map().span_until_char(stmt.span.source_callsite(), '!'), "_");
            if let Some(callee) = stmt.span.source_callee() {
                println!("EXPR {:#?}", name);
            }
        }
    }
    fn check_pat(&mut self, ecx: &EarlyContext<'_>, pat: &ast::Pat) {
        if in_macro(pat.span) {
            let name = snippet(ecx, ecx.sess.source_map().span_until_char(pat.span.source_callsite(), '!'), "_");
            if let Some(callee) = pat.span.source_callee() {
                println!("EXPR {:#?}", name);
            }
        }
    }
}

fn find_used_macros(ecx: &EarlyContext<'_>, path: &str) {
    for it in ecx.krate.module.items.iter() {
        if in_macro(it.span) {
            // println!("{:#?}", it)
        }
    }
    for x in ecx.sess.imported_macro_spans.borrow().iter() {
        // println!("{:?}", x);
    }
}
