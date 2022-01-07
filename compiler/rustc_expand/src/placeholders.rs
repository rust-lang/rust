use crate::expand::{AstFragment, AstFragmentKind};

use rustc_ast as ast;
use rustc_ast::mut_visit::*;
use rustc_ast::ptr::P;
use rustc_span::source_map::DUMMY_SP;
use rustc_span::symbol::Ident;

use smallvec::{smallvec, SmallVec};

use rustc_data_structures::fx::FxHashMap;

pub fn placeholder(
    kind: AstFragmentKind,
    id: ast::NodeId,
    vis: Option<ast::Visibility>,
) -> AstFragment {
    fn mac_placeholder() -> ast::MacCall {
        ast::MacCall {
            path: ast::Path { span: DUMMY_SP, segments: Vec::new(), tokens: None },
            args: P(ast::MacArgs::Empty),
            prior_type_ascription: None,
        }
    }

    let ident = Ident::empty();
    let attrs = Vec::new();
    let vis = vis.unwrap_or(ast::Visibility {
        span: DUMMY_SP,
        kind: ast::VisibilityKind::Inherited,
        tokens: None,
    });
    let span = DUMMY_SP;
    let expr_placeholder = || {
        P(ast::Expr {
            id,
            span,
            attrs: ast::AttrVec::new(),
            kind: ast::ExprKind::MacCall(mac_placeholder()),
            tokens: None,
        })
    };
    let ty =
        || P(ast::Ty { id, kind: ast::TyKind::MacCall(mac_placeholder()), span, tokens: None });
    let pat =
        || P(ast::Pat { id, kind: ast::PatKind::MacCall(mac_placeholder()), span, tokens: None });

    match kind {
        AstFragmentKind::Crate => AstFragment::Crate(ast::Crate {
            attrs: Default::default(),
            items: Default::default(),
            span,
            id,
            is_placeholder: true,
        }),
        AstFragmentKind::Expr => AstFragment::Expr(expr_placeholder()),
        AstFragmentKind::OptExpr => AstFragment::OptExpr(Some(expr_placeholder())),
        AstFragmentKind::Items => AstFragment::Items(smallvec![P(ast::Item {
            id,
            span,
            ident,
            vis,
            attrs,
            kind: ast::ItemKind::MacCall(mac_placeholder()),
            tokens: None,
        })]),
        AstFragmentKind::TraitItems => AstFragment::TraitItems(smallvec![P(ast::AssocItem {
            id,
            span,
            ident,
            vis,
            attrs,
            kind: ast::AssocItemKind::MacCall(mac_placeholder()),
            tokens: None,
        })]),
        AstFragmentKind::ImplItems => AstFragment::ImplItems(smallvec![P(ast::AssocItem {
            id,
            span,
            ident,
            vis,
            attrs,
            kind: ast::AssocItemKind::MacCall(mac_placeholder()),
            tokens: None,
        })]),
        AstFragmentKind::ForeignItems => {
            AstFragment::ForeignItems(smallvec![P(ast::ForeignItem {
                id,
                span,
                ident,
                vis,
                attrs,
                kind: ast::ForeignItemKind::MacCall(mac_placeholder()),
                tokens: None,
            })])
        }
        AstFragmentKind::Pat => AstFragment::Pat(P(ast::Pat {
            id,
            span,
            kind: ast::PatKind::MacCall(mac_placeholder()),
            tokens: None,
        })),
        AstFragmentKind::Ty => AstFragment::Ty(P(ast::Ty {
            id,
            span,
            kind: ast::TyKind::MacCall(mac_placeholder()),
            tokens: None,
        })),
        AstFragmentKind::Stmts => AstFragment::Stmts(smallvec![{
            let mac = P(ast::MacCallStmt {
                mac: mac_placeholder(),
                style: ast::MacStmtStyle::Braces,
                attrs: ast::AttrVec::new(),
                tokens: None,
            });
            ast::Stmt { id, span, kind: ast::StmtKind::MacCall(mac) }
        }]),
        AstFragmentKind::Arms => AstFragment::Arms(smallvec![ast::Arm {
            attrs: Default::default(),
            body: expr_placeholder(),
            guard: None,
            id,
            pat: pat(),
            span,
            is_placeholder: true,
        }]),
        AstFragmentKind::Fields => AstFragment::Fields(smallvec![ast::ExprField {
            attrs: Default::default(),
            expr: expr_placeholder(),
            id,
            ident,
            is_shorthand: false,
            span,
            is_placeholder: true,
        }]),
        AstFragmentKind::FieldPats => AstFragment::FieldPats(smallvec![ast::PatField {
            attrs: Default::default(),
            id,
            ident,
            is_shorthand: false,
            pat: pat(),
            span,
            is_placeholder: true,
        }]),
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
        AstFragmentKind::Params => AstFragment::Params(smallvec![ast::Param {
            attrs: Default::default(),
            id,
            pat: pat(),
            span,
            ty: ty(),
            is_placeholder: true,
        }]),
        AstFragmentKind::StructFields => AstFragment::StructFields(smallvec![ast::FieldDef {
            attrs: Default::default(),
            id,
            ident: None,
            span,
            ty: ty(),
            vis,
            is_placeholder: true,
        }]),
        AstFragmentKind::Variants => AstFragment::Variants(smallvec![ast::Variant {
            attrs: Default::default(),
            data: ast::VariantData::Struct(Default::default(), false),
            disr_expr: None,
            id,
            ident,
            span,
            vis,
            is_placeholder: true,
        }]),
    }
}

#[derive(Default)]
pub struct PlaceholderExpander {
    expanded_fragments: FxHashMap<ast::NodeId, AstFragment>,
}

impl PlaceholderExpander {
    pub fn add(&mut self, id: ast::NodeId, mut fragment: AstFragment) {
        fragment.mut_visit_with(self);
        self.expanded_fragments.insert(id, fragment);
    }

    fn remove(&mut self, id: ast::NodeId) -> AstFragment {
        self.expanded_fragments.remove(&id).unwrap()
    }
}

impl MutVisitor for PlaceholderExpander {
    fn flat_map_arm(&mut self, arm: ast::Arm) -> SmallVec<[ast::Arm; 1]> {
        if arm.is_placeholder {
            self.remove(arm.id).make_arms()
        } else {
            noop_flat_map_arm(arm, self)
        }
    }

    fn flat_map_expr_field(&mut self, field: ast::ExprField) -> SmallVec<[ast::ExprField; 1]> {
        if field.is_placeholder {
            self.remove(field.id).make_expr_fields()
        } else {
            noop_flat_map_expr_field(field, self)
        }
    }

    fn flat_map_pat_field(&mut self, fp: ast::PatField) -> SmallVec<[ast::PatField; 1]> {
        if fp.is_placeholder {
            self.remove(fp.id).make_pat_fields()
        } else {
            noop_flat_map_pat_field(fp, self)
        }
    }

    fn flat_map_generic_param(
        &mut self,
        param: ast::GenericParam,
    ) -> SmallVec<[ast::GenericParam; 1]> {
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

    fn flat_map_field_def(&mut self, sf: ast::FieldDef) -> SmallVec<[ast::FieldDef; 1]> {
        if sf.is_placeholder {
            self.remove(sf.id).make_field_defs()
        } else {
            noop_flat_map_field_def(sf, self)
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
            ast::ItemKind::MacCall(_) => self.remove(item.id).make_items(),
            _ => noop_flat_map_item(item, self),
        }
    }

    fn flat_map_trait_item(&mut self, item: P<ast::AssocItem>) -> SmallVec<[P<ast::AssocItem>; 1]> {
        match item.kind {
            ast::AssocItemKind::MacCall(_) => self.remove(item.id).make_trait_items(),
            _ => noop_flat_map_assoc_item(item, self),
        }
    }

    fn flat_map_impl_item(&mut self, item: P<ast::AssocItem>) -> SmallVec<[P<ast::AssocItem>; 1]> {
        match item.kind {
            ast::AssocItemKind::MacCall(_) => self.remove(item.id).make_impl_items(),
            _ => noop_flat_map_assoc_item(item, self),
        }
    }

    fn flat_map_foreign_item(
        &mut self,
        item: P<ast::ForeignItem>,
    ) -> SmallVec<[P<ast::ForeignItem>; 1]> {
        match item.kind {
            ast::ForeignItemKind::MacCall(_) => self.remove(item.id).make_foreign_items(),
            _ => noop_flat_map_foreign_item(item, self),
        }
    }

    fn visit_expr(&mut self, expr: &mut P<ast::Expr>) {
        match expr.kind {
            ast::ExprKind::MacCall(_) => *expr = self.remove(expr.id).make_expr(),
            _ => noop_visit_expr(expr, self),
        }
    }

    fn filter_map_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        match expr.kind {
            ast::ExprKind::MacCall(_) => self.remove(expr.id).make_opt_expr(),
            _ => noop_filter_map_expr(expr, self),
        }
    }

    fn flat_map_stmt(&mut self, stmt: ast::Stmt) -> SmallVec<[ast::Stmt; 1]> {
        let (style, mut stmts) = match stmt.kind {
            ast::StmtKind::MacCall(mac) => (mac.style, self.remove(stmt.id).make_stmts()),
            _ => return noop_flat_map_stmt(stmt, self),
        };

        if style == ast::MacStmtStyle::Semicolon {
            // Implement the proposal described in
            // https://github.com/rust-lang/rust/issues/61733#issuecomment-509626449
            //
            // The macro invocation expands to the list of statements. If the
            // list of statements is empty, then 'parse' the trailing semicolon
            // on the original invocation as an empty statement. That is:
            //
            // `empty();` is parsed as a single `StmtKind::Empty`
            //
            // If the list of statements is non-empty, see if the final
            // statement already has a trailing semicolon.
            //
            // If it doesn't have a semicolon, then 'parse' the trailing
            // semicolon from the invocation as part of the final statement,
            // using `stmt.add_trailing_semicolon()`
            //
            // If it does have a semicolon, then 'parse' the trailing semicolon
            // from the invocation as a new StmtKind::Empty

            // FIXME: We will need to preserve the original semicolon token and
            // span as part of #15701
            let empty_stmt =
                ast::Stmt { id: ast::DUMMY_NODE_ID, kind: ast::StmtKind::Empty, span: DUMMY_SP };

            if let Some(stmt) = stmts.pop() {
                if stmt.has_trailing_semicolon() {
                    stmts.push(stmt);
                    stmts.push(empty_stmt);
                } else {
                    stmts.push(stmt.add_trailing_semicolon());
                }
            } else {
                stmts.push(empty_stmt);
            }
        }

        stmts
    }

    fn visit_pat(&mut self, pat: &mut P<ast::Pat>) {
        match pat.kind {
            ast::PatKind::MacCall(_) => *pat = self.remove(pat.id).make_pat(),
            _ => noop_visit_pat(pat, self),
        }
    }

    fn visit_ty(&mut self, ty: &mut P<ast::Ty>) {
        match ty.kind {
            ast::TyKind::MacCall(_) => *ty = self.remove(ty.id).make_ty(),
            _ => noop_visit_ty(ty, self),
        }
    }

    fn visit_crate(&mut self, krate: &mut ast::Crate) {
        if krate.is_placeholder {
            *krate = self.remove(krate.id).make_crate();
        } else {
            noop_visit_crate(krate, self)
        }
    }
}
