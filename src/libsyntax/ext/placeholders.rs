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
use ext::base::ExtCtxt;
use ext::expand::{Expansion, ExpansionKind};
use ext::hygiene::Mark;
use fold::*;
use ptr::P;
use symbol::keywords;
use util::move_map::MoveMap;
use util::small_vector::SmallVector;

use std::collections::HashMap;
use std::mem;

pub fn placeholder(kind: ExpansionKind, id: ast::NodeId) -> Expansion {
    fn mac_placeholder() -> ast::Mac {
        dummy_spanned(ast::Mac_ {
            path: ast::Path { span: DUMMY_SP, segments: Vec::new() },
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

pub struct PlaceholderExpander<'a, 'b: 'a> {
    expansions: HashMap<ast::NodeId, Expansion>,
    cx: &'a mut ExtCtxt<'b>,
    monotonic: bool,
}

impl<'a, 'b> PlaceholderExpander<'a, 'b> {
    pub fn new(cx: &'a mut ExtCtxt<'b>, monotonic: bool) -> Self {
        PlaceholderExpander {
            cx: cx,
            expansions: HashMap::new(),
            monotonic: monotonic,
        }
    }

    pub fn add(&mut self, id: ast::NodeId, expansion: Expansion) {
        let expansion = expansion.fold_with(self);
        self.expansions.insert(id, expansion);
    }

    fn remove(&mut self, id: ast::NodeId) -> Expansion {
        self.expansions.remove(&id).unwrap()
    }
}

impl<'a, 'b> Folder for PlaceholderExpander<'a, 'b> {
    fn fold_item(&mut self, item: P<ast::Item>) -> SmallVector<P<ast::Item>> {
        match item.node {
            ast::ItemKind::Mac(ref mac) if !mac.node.path.segments.is_empty() => {}
            ast::ItemKind::Mac(_) => return self.remove(item.id).make_items(),
            _ => {}
        }

        noop_fold_item(item, self)
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

    fn fold_block(&mut self, block: P<ast::Block>) -> P<ast::Block> {
        noop_fold_block(block, self).map(|mut block| {
            let mut macros = Vec::new();
            let mut remaining_stmts = block.stmts.len();

            block.stmts = block.stmts.move_flat_map(|mut stmt| {
                remaining_stmts -= 1;

                // `macro_rules!` macro definition
                if let ast::StmtKind::Item(ref item) = stmt.node {
                    if let ast::ItemKind::Mac(_) = item.node {
                        macros.push(Mark::from_placeholder_id(item.id));
                        return None;
                    }
                }

                match stmt.node {
                    // Avoid wasting a node id on a trailing expression statement,
                    // which shares a HIR node with the expression itself.
                    ast::StmtKind::Expr(ref expr) if remaining_stmts == 0 => stmt.id = expr.id,

                    _ if self.monotonic => {
                        assert_eq!(stmt.id, ast::DUMMY_NODE_ID);
                        stmt.id = self.cx.resolver.next_node_id();
                    }

                    _ => {}
                }

                if self.monotonic && !macros.is_empty() {
                    let macros = mem::replace(&mut macros, Vec::new());
                    self.cx.resolver.add_expansions_at_stmt(stmt.id, macros);
                }

                Some(stmt)
            });

            block
        })
    }

    fn fold_mod(&mut self, module: ast::Mod) -> ast::Mod {
        let mut module = noop_fold_mod(module, self);
        module.items = module.items.move_flat_map(|item| match item.node {
            ast::ItemKind::Mac(_) if !self.cx.ecfg.keep_macs => None, // remove macro definitions
            _ => Some(item),
        });
        module
    }

    fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac {
        mac
    }
}
