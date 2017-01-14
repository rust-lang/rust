// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty;
use rustc::ty::adjustment;
use util::nodemap::FxHashMap;
use lint::{LateContext, EarlyContext, LintContext, LintArray};
use lint::{LintPass, EarlyLintPass, LateLintPass};

use std::collections::hash_map::Entry::{Occupied, Vacant};

use syntax::ast;
use syntax::attr;
use syntax::feature_gate::{BUILTIN_ATTRIBUTES, AttributeType};
use syntax::symbol::keywords;
use syntax::ptr::P;
use syntax_pos::Span;

use rustc_back::slice;
use rustc::hir;
use rustc::hir::intravisit::FnKind;

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

        let mut mutables = FxHashMap();
        for p in pats {
            p.each_binding(|mode, id, _, path1| {
                let name = path1.node;
                if let hir::BindByValue(hir::MutMutable) = mode {
                    if !name.as_str().starts_with("_") {
                        match mutables.entry(name) {
                            Vacant(entry) => {
                                entry.insert(vec![id]);
                            }
                            Occupied(mut entry) => {
                                entry.get_mut().push(id);
                            }
                        }
                    }
                }
            });
        }

        let used_mutables = cx.tcx.used_mut_nodes.borrow();
        for (_, v) in &mutables {
            if !v.iter().any(|e| used_mutables.contains(e)) {
                cx.span_lint(UNUSED_MUT,
                             cx.tcx.map.span(v[0]),
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

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnusedMut {
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

    fn check_fn(&mut self,
                cx: &LateContext,
                _: FnKind,
                _: &hir::FnDecl,
                body: &hir::Body,
                _: Span,
                _: ast::NodeId) {
        for a in &body.arguments {
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

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnusedResults {
    fn check_stmt(&mut self, cx: &LateContext, s: &hir::Stmt) {
        let expr = match s.node {
            hir::StmtSemi(ref expr, _) => &**expr,
            _ => return,
        };

        if let hir::ExprRet(..) = expr.node {
            return;
        }

        let t = cx.tables.expr_ty(&expr);
        let warned = match t.sty {
            ty::TyTuple(ref tys) if tys.is_empty() => return,
            ty::TyNever => return,
            ty::TyBool => return,
            ty::TyAdt(def, _) => {
                let attrs = cx.tcx.get_attrs(def.did);
                check_must_use(cx, &attrs[..], s.span)
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
                    if let Some(s) = attr.value_str() {
                        msg.push_str(": ");
                        msg.push_str(&s.as_str());
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

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnusedUnsafe {
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

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for PathStatements {
    fn check_stmt(&mut self, cx: &LateContext, s: &hir::Stmt) {
        if let hir::StmtSemi(ref expr, _) = s.node {
            if let hir::ExprPath(_) = expr.node {
                cx.span_lint(PATH_STATEMENTS, s.span, "path statement with no effect");
            }
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

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnusedAttributes {
    fn check_attribute(&mut self, cx: &LateContext, attr: &ast::Attribute) {
        debug!("checking attribute: {:?}", attr);

        // Note that check_name() marks the attribute as used if it matches.
        for &(ref name, ty, _) in BUILTIN_ATTRIBUTES {
            match ty {
                AttributeType::Whitelisted if attr.check_name(name) => {
                    debug!("{:?} is Whitelisted", name);
                    break;
                }
                _ => (),
            }
        }

        let plugin_attributes = cx.sess().plugin_attributes.borrow_mut();
        for &(ref name, ty) in plugin_attributes.iter() {
            if ty == AttributeType::Whitelisted && attr.check_name(&name) {
                debug!("{:?} (plugin attr) is whitelisted with ty {:?}", name, ty);
                break;
            }
        }

        if !attr::is_used(attr) {
            debug!("Emitting warning for: {:?}", attr);
            cx.span_lint(UNUSED_ATTRIBUTES, attr.span, "unused attribute");
            // Is it a builtin attribute that must be used at the crate level?
            let known_crate = BUILTIN_ATTRIBUTES.iter()
                .find(|&&(name, ty, _)| attr.name() == name && ty == AttributeType::CrateLevel)
                .is_some();

            // Has a plugin registered this attribute as one which must be used at
            // the crate level?
            let plugin_crate = plugin_attributes.iter()
                .find(|&&(ref x, t)| attr.name() == &**x && AttributeType::CrateLevel == t)
                .is_some();
            if known_crate || plugin_crate {
                let msg = match attr.style {
                    ast::AttrStyle::Outer => {
                        "crate-level attribute should be an inner attribute: add an exclamation \
                         mark: #![foo]"
                    }
                    ast::AttrStyle::Inner => "crate-level attribute should be in the root module",
                };
                cx.span_lint(UNUSED_ATTRIBUTES, attr.span, msg);
            }
        } else {
            debug!("Attr was used: {:?}", attr);
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
    fn check_unused_parens_core(&self,
                                cx: &EarlyContext,
                                value: &ast::Expr,
                                msg: &str,
                                struct_lit_needs_parens: bool) {
        if let ast::ExprKind::Paren(ref inner) = value.node {
            let necessary = struct_lit_needs_parens && contains_exterior_struct_lit(&inner);
            if !necessary {
                cx.span_lint(UNUSED_PARENS,
                             value.span,
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
                ast::ExprKind::Struct(..) => true,

                ast::ExprKind::Assign(ref lhs, ref rhs) |
                ast::ExprKind::AssignOp(_, ref lhs, ref rhs) |
                ast::ExprKind::Binary(_, ref lhs, ref rhs) => {
                    // X { y: 1 } + X { y: 2 }
                    contains_exterior_struct_lit(&lhs) || contains_exterior_struct_lit(&rhs)
                }
                ast::ExprKind::Unary(_, ref x) |
                ast::ExprKind::Cast(ref x, _) |
                ast::ExprKind::Type(ref x, _) |
                ast::ExprKind::Field(ref x, _) |
                ast::ExprKind::TupField(ref x, _) |
                ast::ExprKind::Index(ref x, _) => {
                    // &X { y: 1 }, X { y: 1 }.y
                    contains_exterior_struct_lit(&x)
                }

                ast::ExprKind::MethodCall(.., ref exprs) => {
                    // X { y: 1 }.bar(...)
                    contains_exterior_struct_lit(&exprs[0])
                }

                _ => false,
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
        use syntax::ast::ExprKind::*;
        let (value, msg, struct_lit_needs_parens) = match e.node {
            If(ref cond, ..) => (cond, "`if` condition", true),
            While(ref cond, ..) => (cond, "`while` condition", true),
            IfLet(_, ref cond, ..) => (cond, "`if let` head expression", true),
            WhileLet(_, ref cond, ..) => (cond, "`while let` head expression", true),
            ForLoop(_, ref cond, ..) => (cond, "`for` head expression", true),
            Match(ref head, _) => (head, "`match` head expression", true),
            Ret(Some(ref value)) => (value, "`return` value", false),
            Assign(_, ref value) => (value, "assigned value", false),
            AssignOp(.., ref value) => (value, "assigned value", false),
            InPlace(_, ref value) => (value, "emplacement value", false),
            _ => return,
        };
        self.check_unused_parens_core(cx, &value, msg, struct_lit_needs_parens);
    }

    fn check_stmt(&mut self, cx: &EarlyContext, s: &ast::Stmt) {
        let (value, msg) = match s.node {
            ast::StmtKind::Local(ref local) => {
                match local.init {
                    Some(ref value) => (value, "assigned value"),
                    None => return,
                }
            }
            _ => return,
        };
        self.check_unused_parens_core(cx, &value, msg, false);
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

impl EarlyLintPass for UnusedImportBraces {
    fn check_item(&mut self, cx: &EarlyContext, item: &ast::Item) {
        if let ast::ItemKind::Use(ref view_path) = item.node {
            if let ast::ViewPathList(_, ref items) = view_path.node {
                if items.len() == 1 && items[0].node.name.name != keywords::SelfValue.name() {
                    let msg = format!("braces around {} is unnecessary", items[0].node.name);
                    cx.span_lint(UNUSED_IMPORT_BRACES, item.span, &msg);
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

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnusedAllocation {
    fn check_expr(&mut self, cx: &LateContext, e: &hir::Expr) {
        match e.node {
            hir::ExprBox(_) => {}
            _ => return,
        }

        if let Some(adjustment) = cx.tables.adjustments.get(&e.id) {
            if let adjustment::Adjust::DerefRef { autoref, .. } = adjustment.kind {
                match autoref {
                    Some(adjustment::AutoBorrow::Ref(_, hir::MutImmutable)) => {
                        cx.span_lint(UNUSED_ALLOCATION,
                                     e.span,
                                     "unnecessary allocation, use & instead");
                    }
                    Some(adjustment::AutoBorrow::Ref(_, hir::MutMutable)) => {
                        cx.span_lint(UNUSED_ALLOCATION,
                                     e.span,
                                     "unnecessary allocation, use &mut instead");
                    }
                    _ => (),
                }
            }
        }
    }
}
