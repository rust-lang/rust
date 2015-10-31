// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use metadata::csearch;
use middle::pat_util;
use middle::ty;
use middle::ty::adjustment;
use rustc::front::map as hir_map;
use util::nodemap::FnvHashMap;
use lint::{LateContext, EarlyContext, LintContext, LintArray};
use lint::{LintPass, EarlyLintPass, LateLintPass};

use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::slice;

use syntax::ast;
use syntax::attr::{self, AttrMetaMethods};
use syntax::codemap::Span;
use syntax::feature_gate::{KNOWN_ATTRIBUTES, AttributeType};
use syntax::ptr::P;

use rustc_front::hir;

use rustc_front::visit::FnKind;

declare_lint! {
    pub UNUSED_MUT,
    Warn,
    "detect mut variables which don't need to be mutable"
}

#[derive(Copy, Clone)]
pub struct UnusedMut;

impl UnusedMut {
    fn check_unused_mut_pat(&self, cx: &LateContext, pats: &[P<hir::Pat>]) {
        // collect all mutable pattern and group their NodeIDs by their Identifier to
        // avoid false warnings in match arms with multiple patterns

        let mut mutables = FnvHashMap();
        for p in pats {
            pat_util::pat_bindings(&cx.tcx.def_map, p, |mode, id, _, path1| {
                let name = path1.node;
                if let hir::BindByValue(hir::MutMutable) = mode {
                    if !name.as_str().starts_with("_") {
                        match mutables.entry(name.0 as usize) {
                            Vacant(entry) => { entry.insert(vec![id]); },
                            Occupied(mut entry) => { entry.get_mut().push(id); },
                        }
                    }
                }
            });
        }

        let used_mutables = cx.tcx.used_mut_nodes.borrow();
        for (_, v) in &mutables {
            if !v.iter().any(|e| used_mutables.contains(e)) {
                cx.span_lint(UNUSED_MUT, cx.tcx.map.span(v[0]),
                             "variable does not need to be mutable");
            }
        }
    }
}

impl LintPass for UnusedMut {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNUSED_MUT)
    }
}

impl LateLintPass for UnusedMut {
    fn check_expr(&mut self, cx: &LateContext, e: &hir::Expr) {
        if let hir::ExprMatch(_, ref arms, _) = e.node {
            for a in arms {
                self.check_unused_mut_pat(cx, &a.pats)
            }
        }
    }

    fn check_stmt(&mut self, cx: &LateContext, s: &hir::Stmt) {
        if let hir::StmtDecl(ref d, _) = s.node {
            if let hir::DeclLocal(ref l) = d.node {
                self.check_unused_mut_pat(cx, slice::ref_slice(&l.pat));
            }
        }
    }

    fn check_fn(&mut self, cx: &LateContext,
                _: FnKind, decl: &hir::FnDecl,
                _: &hir::Block, _: Span, _: ast::NodeId) {
        for a in &decl.inputs {
            self.check_unused_mut_pat(cx, slice::ref_slice(&a.pat));
        }
    }
}

declare_lint! {
    pub UNUSED_MUST_USE,
    Warn,
    "unused result of a type flagged as #[must_use]"
}

declare_lint! {
    pub UNUSED_RESULTS,
    Allow,
    "unused result of an expression in a statement"
}

#[derive(Copy, Clone)]
pub struct UnusedResults;

impl LintPass for UnusedResults {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNUSED_MUST_USE, UNUSED_RESULTS)
    }
}

impl LateLintPass for UnusedResults {
    fn check_stmt(&mut self, cx: &LateContext, s: &hir::Stmt) {
        let expr = match s.node {
            hir::StmtSemi(ref expr, _) => &**expr,
            _ => return
        };

        if let hir::ExprRet(..) = expr.node {
            return;
        }

        let t = cx.tcx.expr_ty(&expr);
        let warned = match t.sty {
            ty::TyTuple(ref tys) if tys.is_empty() => return,
            ty::TyBool => return,
            ty::TyStruct(def, _) |
            ty::TyEnum(def, _) => {
                if let Some(def_node_id) = cx.tcx.map.as_local_node_id(def.did) {
                    if let hir_map::NodeItem(it) = cx.tcx.map.get(def_node_id) {
                        check_must_use(cx, &it.attrs, s.span)
                    } else {
                        false
                    }
                } else {
                    let attrs = csearch::get_item_attrs(&cx.sess().cstore, def.did);
                    check_must_use(cx, &attrs[..], s.span)
                }
            }
            _ => false,
        };
        if !warned {
            cx.span_lint(UNUSED_RESULTS, s.span, "unused result");
        }

        fn check_must_use(cx: &LateContext, attrs: &[ast::Attribute], sp: Span) -> bool {
            for attr in attrs {
                if attr.check_name("must_use") {
                    let mut msg = "unused result which must be used".to_string();
                    // check for #[must_use="..."]
                    match attr.value_str() {
                        None => {}
                        Some(s) => {
                            msg.push_str(": ");
                            msg.push_str(&s);
                        }
                    }
                    cx.span_lint(UNUSED_MUST_USE, sp, &msg);
                    return true;
                }
            }
            false
        }
    }
}

declare_lint! {
    pub UNUSED_UNSAFE,
    Warn,
    "unnecessary use of an `unsafe` block"
}

#[derive(Copy, Clone)]
pub struct UnusedUnsafe;

impl LintPass for UnusedUnsafe {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNUSED_UNSAFE)
    }
}

impl LateLintPass for UnusedUnsafe {
    fn check_expr(&mut self, cx: &LateContext, e: &hir::Expr) {
        if let hir::ExprBlock(ref blk) = e.node {
            // Don't warn about generated blocks, that'll just pollute the output.
            if blk.rules == hir::UnsafeBlock(hir::UserProvided) &&
                !cx.tcx.used_unsafe.borrow().contains(&blk.id) {
                    cx.span_lint(UNUSED_UNSAFE, blk.span, "unnecessary `unsafe` block");
            }
        }
    }
}

declare_lint! {
    pub PATH_STATEMENTS,
    Warn,
    "path statements with no effect"
}

#[derive(Copy, Clone)]
pub struct PathStatements;

impl LintPass for PathStatements {
    fn get_lints(&self) -> LintArray {
        lint_array!(PATH_STATEMENTS)
    }
}

impl LateLintPass for PathStatements {
    fn check_stmt(&mut self, cx: &LateContext, s: &hir::Stmt) {
        match s.node {
            hir::StmtSemi(ref expr, _) => {
                match expr.node {
                    hir::ExprPath(..) => cx.span_lint(PATH_STATEMENTS, s.span,
                                                      "path statement with no effect"),
                    _ => ()
                }
            }
            _ => ()
        }
    }
}

declare_lint! {
    pub UNUSED_ATTRIBUTES,
    Warn,
    "detects attributes that were not used by the compiler"
}

#[derive(Copy, Clone)]
pub struct UnusedAttributes;

impl LintPass for UnusedAttributes {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNUSED_ATTRIBUTES)
    }
}

impl LateLintPass for UnusedAttributes {
    fn check_attribute(&mut self, cx: &LateContext, attr: &ast::Attribute) {
        // Note that check_name() marks the attribute as used if it matches.
        for &(ref name, ty, _) in KNOWN_ATTRIBUTES {
            match ty {
                AttributeType::Whitelisted if attr.check_name(name) => {
                    break;
                },
                _ => ()
            }
        }

        let plugin_attributes = cx.sess().plugin_attributes.borrow_mut();
        for &(ref name, ty) in plugin_attributes.iter() {
            if ty == AttributeType::Whitelisted && attr.check_name(&*name) {
                break;
            }
        }

        if !attr::is_used(attr) {
            cx.span_lint(UNUSED_ATTRIBUTES, attr.span, "unused attribute");
            // Is it a builtin attribute that must be used at the crate level?
            let known_crate = KNOWN_ATTRIBUTES.iter().find(|&&(name, ty, _)| {
                attr.name() == name &&
                ty == AttributeType::CrateLevel
            }).is_some();

            // Has a plugin registered this attribute as one which must be used at
            // the crate level?
            let plugin_crate = plugin_attributes.iter()
                                                .find(|&&(ref x, t)| {
                                                        &*attr.name() == &*x &&
                                                        AttributeType::CrateLevel == t
                                                    }).is_some();
            if  known_crate || plugin_crate {
                let msg = match attr.node.style {
                    ast::AttrStyle::Outer => "crate-level attribute should be an inner \
                                              attribute: add an exclamation mark: #![foo]",
                    ast::AttrStyle::Inner => "crate-level attribute should be in the \
                                              root module",
                };
                cx.span_lint(UNUSED_ATTRIBUTES, attr.span, msg);
            }
        }
    }
}

declare_lint! {
    UNUSED_PARENS,
    Warn,
    "`if`, `match`, `while` and `return` do not need parentheses"
}

#[derive(Copy, Clone)]
pub struct UnusedParens;

impl UnusedParens {
    fn check_unused_parens_core(&self, cx: &EarlyContext, value: &ast::Expr, msg: &str,
                                struct_lit_needs_parens: bool) {
        if let ast::ExprParen(ref inner) = value.node {
            let necessary = struct_lit_needs_parens && contains_exterior_struct_lit(&**inner);
            if !necessary {
                cx.span_lint(UNUSED_PARENS, value.span,
                             &format!("unnecessary parentheses around {}", msg))
            }
        }

        /// Expressions that syntactically contain an "exterior" struct
        /// literal i.e. not surrounded by any parens or other
        /// delimiters, e.g. `X { y: 1 }`, `X { y: 1 }.method()`, `foo
        /// == X { y: 1 }` and `X { y: 1 } == foo` all do, but `(X {
        /// y: 1 }) == foo` does not.
        fn contains_exterior_struct_lit(value: &ast::Expr) -> bool {
            match value.node {
                ast::ExprStruct(..) => true,

                ast::ExprAssign(ref lhs, ref rhs) |
                ast::ExprAssignOp(_, ref lhs, ref rhs) |
                ast::ExprBinary(_, ref lhs, ref rhs) => {
                    // X { y: 1 } + X { y: 2 }
                    contains_exterior_struct_lit(&**lhs) ||
                        contains_exterior_struct_lit(&**rhs)
                }
                ast::ExprUnary(_, ref x) |
                ast::ExprCast(ref x, _) |
                ast::ExprField(ref x, _) |
                ast::ExprTupField(ref x, _) |
                ast::ExprIndex(ref x, _) => {
                    // &X { y: 1 }, X { y: 1 }.y
                    contains_exterior_struct_lit(&**x)
                }

                ast::ExprMethodCall(_, _, ref exprs) => {
                    // X { y: 1 }.bar(...)
                    contains_exterior_struct_lit(&*exprs[0])
                }

                _ => false
            }
        }
    }
}

impl LintPass for UnusedParens {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNUSED_PARENS)
    }
}

impl EarlyLintPass for UnusedParens {
    fn check_expr(&mut self, cx: &EarlyContext, e: &ast::Expr) {
        let (value, msg, struct_lit_needs_parens) = match e.node {
            ast::ExprIf(ref cond, _, _) => (cond, "`if` condition", true),
            ast::ExprWhile(ref cond, _, _) => (cond, "`while` condition", true),
            ast::ExprIfLet(_, ref cond, _, _) => (cond, "`if let` head expression", true),
            ast::ExprWhileLet(_, ref cond, _, _) => (cond, "`while let` head expression", true),
            ast::ExprForLoop(_, ref cond, _, _) => (cond, "`for` head expression", true),
            ast::ExprMatch(ref head, _) => (head, "`match` head expression", true),
            ast::ExprRet(Some(ref value)) => (value, "`return` value", false),
            ast::ExprAssign(_, ref value) => (value, "assigned value", false),
            ast::ExprAssignOp(_, _, ref value) => (value, "assigned value", false),
            _ => return
        };
        self.check_unused_parens_core(cx, &**value, msg, struct_lit_needs_parens);
    }

    fn check_stmt(&mut self, cx: &EarlyContext, s: &ast::Stmt) {
        let (value, msg) = match s.node {
            ast::StmtDecl(ref decl, _) => match decl.node {
                ast::DeclLocal(ref local) => match local.init {
                    Some(ref value) => (value, "assigned value"),
                    None => return
                },
                _ => return
            },
            _ => return
        };
        self.check_unused_parens_core(cx, &**value, msg, false);
    }
}

declare_lint! {
    UNUSED_IMPORT_BRACES,
    Allow,
    "unnecessary braces around an imported item"
}

#[derive(Copy, Clone)]
pub struct UnusedImportBraces;

impl LintPass for UnusedImportBraces {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNUSED_IMPORT_BRACES)
    }
}

impl LateLintPass for UnusedImportBraces {
    fn check_item(&mut self, cx: &LateContext, item: &hir::Item) {
        if let hir::ItemUse(ref view_path) = item.node {
            if let hir::ViewPathList(_, ref items) = view_path.node {
                if items.len() == 1 {
                    if let hir::PathListIdent {ref name, ..} = items[0].node {
                        let m = format!("braces around {} is unnecessary",
                                        name);
                        cx.span_lint(UNUSED_IMPORT_BRACES, item.span,
                                     &m[..]);
                    }
                }
            }
        }
    }
}

declare_lint! {
    UNUSED_ALLOCATION,
    Warn,
    "detects unnecessary allocations that can be eliminated"
}

#[derive(Copy, Clone)]
pub struct UnusedAllocation;

impl LintPass for UnusedAllocation {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNUSED_ALLOCATION)
    }
}

impl LateLintPass for UnusedAllocation {
    fn check_expr(&mut self, cx: &LateContext, e: &hir::Expr) {
        match e.node {
            hir::ExprBox(_) => {}
            _ => return
        }

        if let Some(adjustment) = cx.tcx.tables.borrow().adjustments.get(&e.id) {
            if let adjustment::AdjustDerefRef(adjustment::AutoDerefRef {
                ref autoref, ..
            }) = *adjustment {
                match autoref {
                    &Some(adjustment::AutoPtr(_, hir::MutImmutable)) => {
                        cx.span_lint(UNUSED_ALLOCATION, e.span,
                                     "unnecessary allocation, use & instead");
                    }
                    &Some(adjustment::AutoPtr(_, hir::MutMutable)) => {
                        cx.span_lint(UNUSED_ALLOCATION, e.span,
                                     "unnecessary allocation, use &mut instead");
                    }
                    _ => ()
                }
            }
        }
    }
}

