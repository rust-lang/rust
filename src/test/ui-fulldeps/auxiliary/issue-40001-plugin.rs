#![feature(box_syntax, plugin, plugin_registrar, rustc_private)]
#![crate_type = "dylib"]

extern crate rustc_driver;
extern crate rustc_hir;
#[macro_use] extern crate rustc_lint;
#[macro_use] extern crate rustc_session;
extern crate rustc_span;
extern crate syntax;

use rustc_hir::intravisit;
use rustc_hir as hir;
use rustc_hir::Node;
use rustc_lint::{LateContext, LintPass, LintArray, LateLintPass, LintContext};
use rustc_driver::plugin::Registry;
use rustc_span::source_map;
use syntax::print::pprust;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.lint_store.register_lints(&[&MISSING_WHITELISTED_ATTR]);
    reg.lint_store.register_late_pass(|| box MissingWhitelistedAttrPass);
}

declare_lint! {
    MISSING_WHITELISTED_ATTR,
    Deny,
    "Checks for missing `whitelisted_attr` attribute"
}

declare_lint_pass!(MissingWhitelistedAttrPass => [MISSING_WHITELISTED_ATTR]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for MissingWhitelistedAttrPass {
    fn check_fn(&mut self,
                cx: &LateContext<'a, 'tcx>,
                _: intravisit::FnKind<'tcx>,
                _: &'tcx hir::FnDecl,
                _: &'tcx hir::Body,
                span: source_map::Span,
                id: hir::HirId) {

        let item = match cx.tcx.hir().get(id) {
            Node::Item(item) => item,
            _ => cx.tcx.hir().expect_item(cx.tcx.hir().get_parent_item(id)),
        };

        let whitelisted = |attr| pprust::attribute_to_string(attr).contains("whitelisted_attr");
        if !item.attrs.iter().any(whitelisted) {
            cx.span_lint(MISSING_WHITELISTED_ATTR, span,
                         "Missing 'whitelisted_attr' attribute");
        }
    }
}
