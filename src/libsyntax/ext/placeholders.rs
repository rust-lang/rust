// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use codemap::{DUMMY_SP, dummy_spanned};
use ext::expand::{Expansion, ExpansionKind};
use fold::*;
use parse::token::keywords;
use ptr::P;
use util::small_vector::SmallVector;

use std::collections::HashMap;

pub fn placeholder(kind: ExpansionKind, id: ast::NodeId) -> Expansion {
    fn mac_placeholder() -> ast::Mac {
        dummy_spanned(ast::Mac_ {
            path: ast::Path { span: DUMMY_SP, global: false, segments: Vec::new() },
            tts: Vec::new(),
        })
    }

    let ident = keywords::Invalid.ident();
    let attrs = Vec::new();
    let vis = ast::Visibility::Inherited;
    let span = DUMMY_SP;
    let expr_placeholder = || P(ast::Expr {
        id: id, span: span,
        attrs: ast::ThinVec::new(),
        node: ast::ExprKind::Mac(mac_placeholder()),
    });

    match kind {
        ExpansionKind::Expr => Expansion::Expr(expr_placeholder()),
        ExpansionKind::OptExpr => Expansion::OptExpr(Some(expr_placeholder())),
        ExpansionKind::Items => Expansion::Items(SmallVector::one(P(ast::Item {
            id: id, span: span, ident: ident, vis: vis, attrs: attrs,
            node: ast::ItemKind::Mac(mac_placeholder()),
        }))),
        ExpansionKind::TraitItems => Expansion::TraitItems(SmallVector::one(ast::TraitItem {
            id: id, span: span, ident: ident, attrs: attrs,
            node: ast::TraitItemKind::Macro(mac_placeholder()),
        })),
        ExpansionKind::ImplItems => Expansion::ImplItems(SmallVector::one(ast::ImplItem {
            id: id, span: span, ident: ident, vis: vis, attrs: attrs,
            node: ast::ImplItemKind::Macro(mac_placeholder()),
            defaultness: ast::Defaultness::Final,
        })),
        ExpansionKind::Pat => Expansion::Pat(P(ast::Pat {
            id: id, span: span, node: ast::PatKind::Mac(mac_placeholder()),
        })),
        ExpansionKind::Ty => Expansion::Ty(P(ast::Ty {
            id: id, span: span, node: ast::TyKind::Mac(mac_placeholder()),
        })),
        ExpansionKind::Stmts => Expansion::Stmts(SmallVector::one({
            let mac = P((mac_placeholder(), ast::MacStmtStyle::Braces, ast::ThinVec::new()));
            ast::Stmt { id: id, span: span, node: ast::StmtKind::Mac(mac) }
        })),
    }
}

pub fn macro_scope_placeholder() -> Expansion {
    placeholder(ExpansionKind::Items, ast::DUMMY_NODE_ID)
}

pub struct PlaceholderExpander {
    expansions: HashMap<ast::NodeId, Expansion>,
}

impl PlaceholderExpander {
    pub fn new() -> Self {
        PlaceholderExpander {
            expansions: HashMap::new(),
        }
    }

    pub fn add(&mut self, id: ast::NodeId, expansion: Expansion) {
        self.expansions.insert(id, expansion);
    }

    pub fn remove(&mut self, id: ast::NodeId) -> Expansion {
        self.expansions.remove(&id).unwrap()
    }
}

impl Folder for PlaceholderExpander {
    fn fold_item(&mut self, item: P<ast::Item>) -> SmallVector<P<ast::Item>> {
        match item.node {
            // Scope placeholder
            ast::ItemKind::Mac(_) if item.id == ast::DUMMY_NODE_ID => SmallVector::one(item),
            ast::ItemKind::Mac(_) => self.remove(item.id).make_items(),
            _ => noop_fold_item(item, self),
        }
    }

    fn fold_trait_item(&mut self, item: ast::TraitItem) -> SmallVector<ast::TraitItem> {
        match item.node {
            ast::TraitItemKind::Macro(_) => self.remove(item.id).make_trait_items(),
            _ => noop_fold_trait_item(item, self),
        }
    }

    fn fold_impl_item(&mut self, item: ast::ImplItem) -> SmallVector<ast::ImplItem> {
        match item.node {
            ast::ImplItemKind::Macro(_) => self.remove(item.id).make_impl_items(),
            _ => noop_fold_impl_item(item, self),
        }
    }

    fn fold_expr(&mut self, expr: P<ast::Expr>) -> P<ast::Expr> {
        match expr.node {
            ast::ExprKind::Mac(_) => self.remove(expr.id).make_expr(),
            _ => expr.map(|expr| noop_fold_expr(expr, self)),
        }
    }

    fn fold_opt_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        match expr.node {
            ast::ExprKind::Mac(_) => self.remove(expr.id).make_opt_expr(),
            _ => noop_fold_opt_expr(expr, self),
        }
    }

    fn fold_stmt(&mut self, stmt: ast::Stmt) -> SmallVector<ast::Stmt> {
        let (style, mut expansion) = match stmt.node {
            ast::StmtKind::Mac(mac) => (mac.1, self.remove(stmt.id).make_stmts()),
            _ => return noop_fold_stmt(stmt, self),
        };

        if style == ast::MacStmtStyle::Semicolon {
            if let Some(stmt) = expansion.pop() {
                expansion.push(stmt.add_trailing_semicolon());
            }
        }

        expansion
    }

    fn fold_pat(&mut self, pat: P<ast::Pat>) -> P<ast::Pat> {
        match pat.node {
            ast::PatKind::Mac(_) => self.remove(pat.id).make_pat(),
            _ => noop_fold_pat(pat, self),
        }
    }

    fn fold_ty(&mut self, ty: P<ast::Ty>) -> P<ast::Ty> {
        match ty.node {
            ast::TyKind::Mac(_) => self.remove(ty.id).make_ty(),
            _ => noop_fold_ty(ty, self),
        }
    }
}

pub fn reconstructed_macro_rules(def: &ast::MacroDef, path: &ast::Path) -> Expansion {
    Expansion::Items(SmallVector::one(P(ast::Item {
        ident: def.ident,
        attrs: def.attrs.clone(),
        id: ast::DUMMY_NODE_ID,
        node: ast::ItemKind::Mac(ast::Mac {
            span: def.span,
            node: ast::Mac_ {
                path: path.clone(),
                tts: def.body.clone(),
            }
        }),
        vis: ast::Visibility::Inherited,
        span: def.span,
    })))
}
