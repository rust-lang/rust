use crate::ast::{self, NodeId};
use crate::source_map::{DUMMY_SP, dummy_spanned};
use crate::ext::base::ExtCtxt;
use crate::ext::expand::{AstFragment, AstFragmentKind};
use crate::ext::hygiene::Mark;
use crate::tokenstream::TokenStream;
use crate::mut_visit::*;
use crate::ptr::P;
use crate::ThinVec;

use smallvec::{smallvec, SmallVec};

use rustc_data_structures::fx::FxHashMap;

pub fn placeholder(kind: AstFragmentKind, id: ast::NodeId) -> AstFragment {
    fn mac_placeholder() -> ast::Mac {
        dummy_spanned(ast::Mac_ {
            path: ast::Path { span: DUMMY_SP, segments: Vec::new() },
            tts: TokenStream::empty().into(),
            delim: ast::MacDelimiter::Brace,
        })
    }

    let ident = ast::Ident::invalid();
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

pub struct PlaceholderExpander<'a, 'b> {
    expanded_fragments: FxHashMap<ast::NodeId, AstFragment>,
    cx: &'a mut ExtCtxt<'b>,
    monotonic: bool,
}

impl<'a, 'b> PlaceholderExpander<'a, 'b> {
    pub fn new(cx: &'a mut ExtCtxt<'b>, monotonic: bool) -> Self {
        PlaceholderExpander {
            cx,
            expanded_fragments: FxHashMap::default(),
            monotonic,
        }
    }

    pub fn add(&mut self, id: ast::NodeId, mut fragment: AstFragment, derives: Vec<Mark>) {
        fragment.mut_visit_with(self);
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

impl<'a, 'b> MutVisitor for PlaceholderExpander<'a, 'b> {
    fn flat_map_item(&mut self, item: P<ast::Item>) -> SmallVec<[P<ast::Item>; 1]> {
        match item.node {
            ast::ItemKind::Mac(_) => return self.remove(item.id).make_items(),
            ast::ItemKind::MacroDef(_) => return smallvec![item],
            _ => {}
        }

        noop_flat_map_item(item, self)
    }

    fn flat_map_trait_item(&mut self, item: ast::TraitItem) -> SmallVec<[ast::TraitItem; 1]> {
        match item.node {
            ast::TraitItemKind::Macro(_) => self.remove(item.id).make_trait_items(),
            _ => noop_flat_map_trait_item(item, self),
        }
    }

    fn flat_map_impl_item(&mut self, item: ast::ImplItem) -> SmallVec<[ast::ImplItem; 1]> {
        match item.node {
            ast::ImplItemKind::Macro(_) => self.remove(item.id).make_impl_items(),
            _ => noop_flat_map_impl_item(item, self),
        }
    }

    fn flat_map_foreign_item(&mut self, item: ast::ForeignItem) -> SmallVec<[ast::ForeignItem; 1]> {
        match item.node {
            ast::ForeignItemKind::Macro(_) => self.remove(item.id).make_foreign_items(),
            _ => noop_flat_map_foreign_item(item, self),
        }
    }

    fn visit_expr(&mut self, expr: &mut P<ast::Expr>) {
        match expr.node {
            ast::ExprKind::Mac(_) => *expr = self.remove(expr.id).make_expr(),
            _ => noop_visit_expr(expr, self),
        }
    }

    fn filter_map_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        match expr.node {
            ast::ExprKind::Mac(_) => self.remove(expr.id).make_opt_expr(),
            _ => noop_filter_map_expr(expr, self),
        }
    }

    fn flat_map_stmt(&mut self, stmt: ast::Stmt) -> SmallVec<[ast::Stmt; 1]> {
        let (style, mut stmts) = match stmt.node {
            ast::StmtKind::Mac(mac) => (mac.1, self.remove(stmt.id).make_stmts()),
            _ => return noop_flat_map_stmt(stmt, self),
        };

        if style == ast::MacStmtStyle::Semicolon {
            if let Some(stmt) = stmts.pop() {
                stmts.push(stmt.add_trailing_semicolon());
            }
        }

        stmts
    }

    fn visit_pat(&mut self, pat: &mut P<ast::Pat>) {
        match pat.node {
            ast::PatKind::Mac(_) => *pat = self.remove(pat.id).make_pat(),
            _ => noop_visit_pat(pat, self),
        }
    }

    fn visit_ty(&mut self, ty: &mut P<ast::Ty>) {
        match ty.node {
            ast::TyKind::Mac(_) => *ty = self.remove(ty.id).make_ty(),
            _ => noop_visit_ty(ty, self),
        }
    }

    fn visit_block(&mut self, block: &mut P<ast::Block>) {
        noop_visit_block(block, self);

        for stmt in block.stmts.iter_mut() {
            if self.monotonic {
                assert_eq!(stmt.id, ast::DUMMY_NODE_ID);
                stmt.id = self.cx.resolver.next_node_id();
            }
        }
    }

    fn visit_mod(&mut self, module: &mut ast::Mod) {
        noop_visit_mod(module, self);
        module.items.retain(|item| match item.node {
            ast::ItemKind::Mac(_) if !self.cx.ecfg.keep_macs => false, // remove macro definitions
            _ => true,
        });
    }

    fn visit_mac(&mut self, _mac: &mut ast::Mac) {
        // Do nothing.
    }
}
