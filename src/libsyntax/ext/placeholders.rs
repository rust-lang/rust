// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{self, NodeId};
use source_map::{DUMMY_SP, dummy_spanned};
use ext::base::ExtCtxt;
use ext::expand::{AstFragment, AstFragmentKind};
use ext::hygiene::Mark;
use tokenstream::TokenStream;
use fold::*;
use ptr::P;
use OneVector;
use symbol::keywords;
use ThinVec;
use util::move_map::MoveMap;

use std::collections::HashMap;

pub fn placeholder(kind: AstFragmentKind, id: ast::NodeId) -> AstFragment {
    fn mac_placeholder() -> ast::Mac {
        dummy_spanned(ast::Mac_ {
            path: ast::Path { span: DUMMY_SP, segments: Vec::new() },
            tts: TokenStream::empty().into(),
            delim: ast::MacDelimiter::Brace,
        })
    }

    let ident = keywords::Invalid.ident();
    let attrs = Vec::new();
    let generics = ast::Generics::default();
    let vis = dummy_spanned(ast::VisibilityKind::Inherited);
    let span = DUMMY_SP;
    let expr_placeholder = || P(ast::Expr {
        id, span,
        attrs: ThinVec::new(),
        node: ast::ExprKind::Mac(mac_placeholder()),
    });

    match kind {
        AstFragmentKind::Expr => AstFragment::Expr(expr_placeholder()),
        AstFragmentKind::OptExpr => AstFragment::OptExpr(Some(expr_placeholder())),
        AstFragmentKind::Items => AstFragment::Items(smallvec![P(ast::Item {
            id, span, ident, vis, attrs,
            node: ast::ItemKind::Mac(mac_placeholder()),
            tokens: None,
        })]),
        AstFragmentKind::TraitItems => AstFragment::TraitItems(smallvec![ast::TraitItem {
            id, span, ident, attrs, generics,
            node: ast::TraitItemKind::Macro(mac_placeholder()),
            tokens: None,
        }]),
        AstFragmentKind::ImplItems => AstFragment::ImplItems(smallvec![ast::ImplItem {
            id, span, ident, vis, attrs, generics,
            node: ast::ImplItemKind::Macro(mac_placeholder()),
            defaultness: ast::Defaultness::Final,
            tokens: None,
        }]),
        AstFragmentKind::ForeignItems =>
            AstFragment::ForeignItems(smallvec![ast::ForeignItem {
                id, span, ident, vis, attrs,
                node: ast::ForeignItemKind::Macro(mac_placeholder()),
            }]),
        AstFragmentKind::Pat => AstFragment::Pat(P(ast::Pat {
            id, span, node: ast::PatKind::Mac(mac_placeholder()),
        })),
        AstFragmentKind::Ty => AstFragment::Ty(P(ast::Ty {
            id, span, node: ast::TyKind::Mac(mac_placeholder()),
        })),
        AstFragmentKind::Stmts => AstFragment::Stmts(smallvec![{
            let mac = P((mac_placeholder(), ast::MacStmtStyle::Braces, ThinVec::new()));
            ast::Stmt { id, span, node: ast::StmtKind::Mac(mac) }
        }]),
    }
}

pub struct PlaceholderExpander<'a, 'b: 'a> {
    expanded_fragments: HashMap<ast::NodeId, AstFragment>,
    cx: &'a mut ExtCtxt<'b>,
    monotonic: bool,
}

impl<'a, 'b> PlaceholderExpander<'a, 'b> {
    pub fn new(cx: &'a mut ExtCtxt<'b>, monotonic: bool) -> Self {
        PlaceholderExpander {
            cx,
            expanded_fragments: HashMap::new(),
            monotonic,
        }
    }

    pub fn add(&mut self, id: ast::NodeId, fragment: AstFragment, derives: Vec<Mark>) {
        let mut fragment = fragment.fold_with(self);
        if let AstFragment::Items(mut items) = fragment {
            for derive in derives {
                match self.remove(NodeId::placeholder_from_mark(derive)) {
                    AstFragment::Items(derived_items) => items.extend(derived_items),
                    _ => unreachable!(),
                }
            }
            fragment = AstFragment::Items(items);
        }
        self.expanded_fragments.insert(id, fragment);
    }

    fn remove(&mut self, id: ast::NodeId) -> AstFragment {
        self.expanded_fragments.remove(&id).unwrap()
    }
}

impl<'a, 'b> Folder for PlaceholderExpander<'a, 'b> {
    fn fold_item(&mut self, item: P<ast::Item>) -> OneVector<P<ast::Item>> {
        match item.node {
            ast::ItemKind::Mac(_) => return self.remove(item.id).make_items(),
            ast::ItemKind::MacroDef(_) => return smallvec![item],
            _ => {}
        }

        noop_fold_item(item, self)
    }

    fn fold_trait_item(&mut self, item: ast::TraitItem) -> OneVector<ast::TraitItem> {
        match item.node {
            ast::TraitItemKind::Macro(_) => self.remove(item.id).make_trait_items(),
            _ => noop_fold_trait_item(item, self),
        }
    }

    fn fold_impl_item(&mut self, item: ast::ImplItem) -> OneVector<ast::ImplItem> {
        match item.node {
            ast::ImplItemKind::Macro(_) => self.remove(item.id).make_impl_items(),
            _ => noop_fold_impl_item(item, self),
        }
    }

    fn fold_foreign_item(&mut self, item: ast::ForeignItem) -> OneVector<ast::ForeignItem> {
        match item.node {
            ast::ForeignItemKind::Macro(_) => self.remove(item.id).make_foreign_items(),
            _ => noop_fold_foreign_item(item, self),
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

    fn fold_stmt(&mut self, stmt: ast::Stmt) -> OneVector<ast::Stmt> {
        let (style, mut stmts) = match stmt.node {
            ast::StmtKind::Mac(mac) => (mac.1, self.remove(stmt.id).make_stmts()),
            _ => return noop_fold_stmt(stmt, self),
        };

        if style == ast::MacStmtStyle::Semicolon {
            if let Some(stmt) = stmts.pop() {
                stmts.push(stmt.add_trailing_semicolon());
            }
        }

        stmts
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
            let mut remaining_stmts = block.stmts.len();

            block.stmts = block.stmts.move_flat_map(|mut stmt| {
                remaining_stmts -= 1;

                if self.monotonic {
                    assert_eq!(stmt.id, ast::DUMMY_NODE_ID);
                    stmt.id = self.cx.resolver.next_node_id();
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
