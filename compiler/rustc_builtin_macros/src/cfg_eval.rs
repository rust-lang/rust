use crate::util::check_builtin_macro_attribute;

use rustc_ast::mut_visit::{self, MutVisitor};
use rustc_ast::ptr::P;
use rustc_ast::{self as ast, AstLike};
use rustc_data_structures::map_in_place::MapInPlace;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_expand::config::StripUnconfigured;
use rustc_expand::configure;
use rustc_span::symbol::sym;
use rustc_span::Span;
use smallvec::SmallVec;

crate fn expand(
    ecx: &mut ExtCtxt<'_>,
    _span: Span,
    meta_item: &ast::MetaItem,
    item: Annotatable,
) -> Vec<Annotatable> {
    check_builtin_macro_attribute(ecx, meta_item, sym::cfg_eval);

    let mut visitor = CfgEval {
        cfg: StripUnconfigured { sess: ecx.sess, features: ecx.ecfg.features, modified: false },
    };
    let mut item = visitor.fully_configure(item);
    if visitor.cfg.modified {
        // Erase the tokens if cfg-stripping modified the item
        // This will cause us to synthesize fake tokens
        // when `nt_to_tokenstream` is called on this item.
        if let Some(tokens) = item.tokens_mut() {
            *tokens = None;
        }
    }
    vec![item]
}

crate struct CfgEval<'a> {
    pub cfg: StripUnconfigured<'a>,
}

impl CfgEval<'_> {
    fn configure<T: AstLike>(&mut self, node: T) -> Option<T> {
        self.cfg.configure(node)
    }

    fn configure_foreign_mod(&mut self, foreign_mod: &mut ast::ForeignMod) {
        let ast::ForeignMod { unsafety: _, abi: _, items } = foreign_mod;
        items.flat_map_in_place(|item| self.configure(item));
    }

    fn configure_variant_data(&mut self, vdata: &mut ast::VariantData) {
        match vdata {
            ast::VariantData::Struct(fields, ..) | ast::VariantData::Tuple(fields, _) => {
                fields.flat_map_in_place(|field| self.configure(field))
            }
            ast::VariantData::Unit(_) => {}
        }
    }

    fn configure_item_kind(&mut self, item: &mut ast::ItemKind) {
        match item {
            ast::ItemKind::Struct(def, _generics) | ast::ItemKind::Union(def, _generics) => {
                self.configure_variant_data(def)
            }
            ast::ItemKind::Enum(ast::EnumDef { variants }, _generics) => {
                variants.flat_map_in_place(|variant| self.configure(variant));
                for variant in variants {
                    self.configure_variant_data(&mut variant.data);
                }
            }
            _ => {}
        }
    }

    fn configure_expr_kind(&mut self, expr_kind: &mut ast::ExprKind) {
        match expr_kind {
            ast::ExprKind::Match(_m, arms) => {
                arms.flat_map_in_place(|arm| self.configure(arm));
            }
            ast::ExprKind::Struct(_path, fields, _base) => {
                fields.flat_map_in_place(|field| self.configure(field));
            }
            _ => {}
        }
    }

    fn configure_pat(&mut self, pat: &mut P<ast::Pat>) {
        if let ast::PatKind::Struct(_path, fields, _etc) = &mut pat.kind {
            fields.flat_map_in_place(|field| self.configure(field));
        }
    }

    fn configure_fn_decl(&mut self, fn_decl: &mut ast::FnDecl) {
        fn_decl.inputs.flat_map_in_place(|arg| self.configure(arg));
    }

    crate fn fully_configure(&mut self, item: Annotatable) -> Annotatable {
        // Since the item itself has already been configured by the InvocationCollector,
        // we know that fold result vector will contain exactly one element
        match item {
            Annotatable::Item(item) => Annotatable::Item(self.flat_map_item(item).pop().unwrap()),
            Annotatable::TraitItem(item) => {
                Annotatable::TraitItem(self.flat_map_trait_item(item).pop().unwrap())
            }
            Annotatable::ImplItem(item) => {
                Annotatable::ImplItem(self.flat_map_impl_item(item).pop().unwrap())
            }
            Annotatable::ForeignItem(item) => {
                Annotatable::ForeignItem(self.flat_map_foreign_item(item).pop().unwrap())
            }
            Annotatable::Stmt(stmt) => {
                Annotatable::Stmt(stmt.map(|stmt| self.flat_map_stmt(stmt).pop().unwrap()))
            }
            Annotatable::Expr(mut expr) => Annotatable::Expr({
                self.visit_expr(&mut expr);
                expr
            }),
            Annotatable::Arm(arm) => Annotatable::Arm(self.flat_map_arm(arm).pop().unwrap()),
            Annotatable::Field(field) => {
                Annotatable::Field(self.flat_map_field(field).pop().unwrap())
            }
            Annotatable::FieldPat(fp) => {
                Annotatable::FieldPat(self.flat_map_field_pattern(fp).pop().unwrap())
            }
            Annotatable::GenericParam(param) => {
                Annotatable::GenericParam(self.flat_map_generic_param(param).pop().unwrap())
            }
            Annotatable::Param(param) => {
                Annotatable::Param(self.flat_map_param(param).pop().unwrap())
            }
            Annotatable::StructField(sf) => {
                Annotatable::StructField(self.flat_map_struct_field(sf).pop().unwrap())
            }
            Annotatable::Variant(v) => {
                Annotatable::Variant(self.flat_map_variant(v).pop().unwrap())
            }
        }
    }
}

impl MutVisitor for CfgEval<'_> {
    fn visit_foreign_mod(&mut self, foreign_mod: &mut ast::ForeignMod) {
        self.configure_foreign_mod(foreign_mod);
        mut_visit::noop_visit_foreign_mod(foreign_mod, self);
    }

    fn visit_item_kind(&mut self, item: &mut ast::ItemKind) {
        self.configure_item_kind(item);
        mut_visit::noop_visit_item_kind(item, self);
    }

    fn visit_expr(&mut self, expr: &mut P<ast::Expr>) {
        self.cfg.configure_expr(expr);
        self.configure_expr_kind(&mut expr.kind);
        mut_visit::noop_visit_expr(expr, self);
    }

    fn filter_map_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        let mut expr = configure!(self, expr);
        self.configure_expr_kind(&mut expr.kind);
        mut_visit::noop_visit_expr(&mut expr, self);
        Some(expr)
    }

    fn flat_map_generic_param(
        &mut self,
        param: ast::GenericParam,
    ) -> SmallVec<[ast::GenericParam; 1]> {
        mut_visit::noop_flat_map_generic_param(configure!(self, param), self)
    }

    fn flat_map_stmt(&mut self, stmt: ast::Stmt) -> SmallVec<[ast::Stmt; 1]> {
        mut_visit::noop_flat_map_stmt(configure!(self, stmt), self)
    }

    fn flat_map_item(&mut self, item: P<ast::Item>) -> SmallVec<[P<ast::Item>; 1]> {
        mut_visit::noop_flat_map_item(configure!(self, item), self)
    }

    fn flat_map_impl_item(&mut self, item: P<ast::AssocItem>) -> SmallVec<[P<ast::AssocItem>; 1]> {
        mut_visit::noop_flat_map_assoc_item(configure!(self, item), self)
    }

    fn flat_map_trait_item(&mut self, item: P<ast::AssocItem>) -> SmallVec<[P<ast::AssocItem>; 1]> {
        mut_visit::noop_flat_map_assoc_item(configure!(self, item), self)
    }

    fn visit_pat(&mut self, pat: &mut P<ast::Pat>) {
        self.configure_pat(pat);
        mut_visit::noop_visit_pat(pat, self)
    }

    fn visit_fn_decl(&mut self, mut fn_decl: &mut P<ast::FnDecl>) {
        self.configure_fn_decl(&mut fn_decl);
        mut_visit::noop_visit_fn_decl(fn_decl, self);
    }
}
