use crate::base::ExtCtxt;
use crate::expand::{AstFragment, AstFragmentKind};

use syntax::ast;
use syntax::source_map::{DUMMY_SP, dummy_spanned};
use syntax::mut_visit::*;
use syntax::ptr::P;
use syntax::ThinVec;

use smallvec::{smallvec, SmallVec};

use rustc_data_structures::fx::FxHashMap;

pub fn placeholder(kind: AstFragmentKind, id: ast::NodeId, vis: Option<ast::Visibility>)
                   -> AstFragment {
    fn mac_placeholder() -> ast::Mac {
        ast::Mac {
            path: ast::Path { span: DUMMY_SP, segments: Vec::new() },
            args: P(ast::MacArgs::Empty),
            prior_type_ascription: None,
        }
    }

    let ident = ast::Ident::invalid();
    let attrs = Vec::new();
    let generics = ast::Generics::default();
    let vis = vis.unwrap_or_else(|| dummy_spanned(ast::VisibilityKind::Inherited));
    let span = DUMMY_SP;
    let expr_placeholder = || P(ast::Expr {
        id, span,
        attrs: ThinVec::new(),
        kind: ast::ExprKind::Mac(mac_placeholder()),
    });
    let ty = || P(ast::Ty {
        id,
        kind: ast::TyKind::Mac(mac_placeholder()),
        span,
    });
    let pat = || P(ast::Pat {
        id,
        kind: ast::PatKind::Mac(mac_placeholder()),
        span,
    });

    match kind {
        AstFragmentKind::Expr => AstFragment::Expr(expr_placeholder()),
        AstFragmentKind::OptExpr => AstFragment::OptExpr(Some(expr_placeholder())),
        AstFragmentKind::Items => AstFragment::Items(smallvec![P(ast::Item {
            id, span, ident, vis, attrs,
            kind: ast::ItemKind::Mac(mac_placeholder()),
            tokens: None,
        })]),
        AstFragmentKind::TraitItems => AstFragment::TraitItems(smallvec![ast::TraitItem {
            id, span, ident, vis, attrs, generics,
            kind: ast::TraitItemKind::Macro(mac_placeholder()),
            tokens: None,
        }]),
        AstFragmentKind::ImplItems => AstFragment::ImplItems(smallvec![ast::ImplItem {
            id, span, ident, vis, attrs, generics,
            kind: ast::ImplItemKind::Macro(mac_placeholder()),
            defaultness: ast::Defaultness::Final,
            tokens: None,
        }]),
        AstFragmentKind::ForeignItems =>
            AstFragment::ForeignItems(smallvec![ast::ForeignItem {
                id, span, ident, vis, attrs,
                kind: ast::ForeignItemKind::Macro(mac_placeholder()),
                tokens: None,
            }]),
        AstFragmentKind::Pat => AstFragment::Pat(P(ast::Pat {
            id, span, kind: ast::PatKind::Mac(mac_placeholder()),
        })),
        AstFragmentKind::Ty => AstFragment::Ty(P(ast::Ty {
            id, span, kind: ast::TyKind::Mac(mac_placeholder()),
        })),
        AstFragmentKind::Stmts => AstFragment::Stmts(smallvec![{
            let mac = P((mac_placeholder(), ast::MacStmtStyle::Braces, ThinVec::new()));
            ast::Stmt { id, span, kind: ast::StmtKind::Mac(mac) }
        }]),
        AstFragmentKind::Arms => AstFragment::Arms(smallvec![
            ast::Arm {
                attrs: Default::default(),
                body: expr_placeholder(),
                guard: None,
                id,
                pat: pat(),
                span,
                is_placeholder: true,
            }
        ]),
        AstFragmentKind::Fields => AstFragment::Fields(smallvec![
            ast::Field {
                attrs: Default::default(),
                expr: expr_placeholder(),
                id,
                ident,
                is_shorthand: false,
                span,
                is_placeholder: true,
            }
        ]),
        AstFragmentKind::FieldPats => AstFragment::FieldPats(smallvec![
            ast::FieldPat {
                attrs: Default::default(),
                id,
                ident,
                is_shorthand: false,
                pat: pat(),
                span,
                is_placeholder: true,
            }
        ]),
        AstFragmentKind::GenericParams => AstFragment::GenericParams(smallvec![{
            ast::GenericParam {
                attrs: Default::default(),
                bounds: Default::default(),
                id,
                ident,
                is_placeholder: true,
                kind: ast::GenericParamKind::Lifetime,
            }
        }]),
        AstFragmentKind::Params => AstFragment::Params(smallvec![
            ast::Param {
                attrs: Default::default(),
                id,
                pat: pat(),
                span,
                ty: ty(),
                is_placeholder: true,
            }
        ]),
        AstFragmentKind::StructFields => AstFragment::StructFields(smallvec![
            ast::StructField {
                attrs: Default::default(),
                id,
                ident: None,
                span,
                ty: ty(),
                vis,
                is_placeholder: true,
            }
        ]),
        AstFragmentKind::Variants => AstFragment::Variants(smallvec![
            ast::Variant {
                attrs: Default::default(),
                data: ast::VariantData::Struct(Default::default(), false),
                disr_expr: None,
                id,
                ident,
                span,
                vis,
                is_placeholder: true,
            }
        ])
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

    pub fn add(&mut self, id: ast::NodeId, mut fragment: AstFragment) {
        fragment.mut_visit_with(self);
        self.expanded_fragments.insert(id, fragment);
    }

    fn remove(&mut self, id: ast::NodeId) -> AstFragment {
        self.expanded_fragments.remove(&id).unwrap()
    }
}

impl<'a, 'b> MutVisitor for PlaceholderExpander<'a, 'b> {
    fn flat_map_arm(&mut self, arm: ast::Arm) -> SmallVec<[ast::Arm; 1]> {
        if arm.is_placeholder {
            self.remove(arm.id).make_arms()
        } else {
            noop_flat_map_arm(arm, self)
        }
    }

    fn flat_map_field(&mut self, field: ast::Field) -> SmallVec<[ast::Field; 1]> {
        if field.is_placeholder {
            self.remove(field.id).make_fields()
        } else {
            noop_flat_map_field(field, self)
        }
    }

    fn flat_map_field_pattern(&mut self, fp: ast::FieldPat) -> SmallVec<[ast::FieldPat; 1]> {
        if fp.is_placeholder {
            self.remove(fp.id).make_field_patterns()
        } else {
            noop_flat_map_field_pattern(fp, self)
        }
    }

    fn flat_map_generic_param(
        &mut self,
        param: ast::GenericParam
    ) -> SmallVec<[ast::GenericParam; 1]>
    {
        if param.is_placeholder {
            self.remove(param.id).make_generic_params()
        } else {
            noop_flat_map_generic_param(param, self)
        }
    }

    fn flat_map_param(&mut self, p: ast::Param) -> SmallVec<[ast::Param; 1]> {
        if p.is_placeholder {
            self.remove(p.id).make_params()
        } else {
            noop_flat_map_param(p, self)
        }
    }

    fn flat_map_struct_field(&mut self, sf: ast::StructField) -> SmallVec<[ast::StructField; 1]> {
        if sf.is_placeholder {
            self.remove(sf.id).make_struct_fields()
        } else {
            noop_flat_map_struct_field(sf, self)
        }
    }

    fn flat_map_variant(&mut self, variant: ast::Variant) -> SmallVec<[ast::Variant; 1]> {
        if variant.is_placeholder {
            self.remove(variant.id).make_variants()
        } else {
            noop_flat_map_variant(variant, self)
        }
    }

    fn flat_map_item(&mut self, item: P<ast::Item>) -> SmallVec<[P<ast::Item>; 1]> {
        match item.kind {
            ast::ItemKind::Mac(_) => return self.remove(item.id).make_items(),
            ast::ItemKind::MacroDef(_) => return smallvec![item],
            _ => {}
        }

        noop_flat_map_item(item, self)
    }

    fn flat_map_trait_item(&mut self, item: ast::TraitItem) -> SmallVec<[ast::TraitItem; 1]> {
        match item.kind {
            ast::TraitItemKind::Macro(_) => self.remove(item.id).make_trait_items(),
            _ => noop_flat_map_trait_item(item, self),
        }
    }

    fn flat_map_impl_item(&mut self, item: ast::ImplItem) -> SmallVec<[ast::ImplItem; 1]> {
        match item.kind {
            ast::ImplItemKind::Macro(_) => self.remove(item.id).make_impl_items(),
            _ => noop_flat_map_impl_item(item, self),
        }
    }

    fn flat_map_foreign_item(&mut self, item: ast::ForeignItem) -> SmallVec<[ast::ForeignItem; 1]> {
        match item.kind {
            ast::ForeignItemKind::Macro(_) => self.remove(item.id).make_foreign_items(),
            _ => noop_flat_map_foreign_item(item, self),
        }
    }

    fn visit_expr(&mut self, expr: &mut P<ast::Expr>) {
        match expr.kind {
            ast::ExprKind::Mac(_) => *expr = self.remove(expr.id).make_expr(),
            _ => noop_visit_expr(expr, self),
        }
    }

    fn filter_map_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        match expr.kind {
            ast::ExprKind::Mac(_) => self.remove(expr.id).make_opt_expr(),
            _ => noop_filter_map_expr(expr, self),
        }
    }

    fn flat_map_stmt(&mut self, stmt: ast::Stmt) -> SmallVec<[ast::Stmt; 1]> {
        let (style, mut stmts) = match stmt.kind {
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
        match pat.kind {
            ast::PatKind::Mac(_) => *pat = self.remove(pat.id).make_pat(),
            _ => noop_visit_pat(pat, self),
        }
    }

    fn visit_ty(&mut self, ty: &mut P<ast::Ty>) {
        match ty.kind {
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
        module.items.retain(|item| match item.kind {
            ast::ItemKind::Mac(_) if !self.cx.ecfg.keep_macs => false, // remove macro definitions
            _ => true,
        });
    }

    fn visit_mac(&mut self, _mac: &mut ast::Mac) {
        // Do nothing.
    }
}
