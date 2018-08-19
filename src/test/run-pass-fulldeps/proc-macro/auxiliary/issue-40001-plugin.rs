// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax, plugin, plugin_registrar, rustc_private)]
#![feature(macro_vis_matcher)]
#![feature(macro_at_most_once_rep)]
#![crate_type = "dylib"]

#[macro_use]
extern crate rustc;
extern crate rustc_plugin;
extern crate syntax;

use rustc_plugin::Registry;
use syntax::attr;
use syntax::ext::base::*;
use syntax::feature_gate::AttributeType::Whitelisted;
use syntax::symbol::Symbol;

use rustc::hir;
use rustc::hir::intravisit;
use rustc::hir::map as hir_map;
use rustc::lint::{LateContext, LintPass, LintArray, LateLintPass, LintContext};
use rustc::ty;
use syntax::{ast, source_map};

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_late_lint_pass(box MissingWhitelistedAttrPass);
    reg.register_attribute("whitelisted_attr".to_string(), Whitelisted);
}

declare_lint!(MISSING_WHITELISTED_ATTR, Deny,
              "Checks for missing `whitelisted_attr` attribute");

struct MissingWhitelistedAttrPass;

impl LintPass for MissingWhitelistedAttrPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(MISSING_WHITELISTED_ATTR)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for MissingWhitelistedAttrPass {
    fn check_fn(&mut self,
                cx: &LateContext<'a, 'tcx>,
                _: intravisit::FnKind<'tcx>,
                _: &'tcx hir::FnDecl,
                _: &'tcx hir::Body,
                span: source_map::Span,
                id: ast::NodeId) {

        let item = match cx.tcx.hir.get(id) {
            hir_map::Node::NodeItem(item) => item,
            _ => cx.tcx.hir.expect_item(cx.tcx.hir.get_parent(id)),
        };

        if !attr::contains_name(&item.attrs, "whitelisted_attr") {
            cx.span_lint(MISSING_WHITELISTED_ATTR, span,
                         "Missing 'whitelisted_attr' attribute");
        }
    }
}
