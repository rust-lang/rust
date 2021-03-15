use crate::util::check_builtin_macro_attribute;

use rustc_ast::mut_visit::{self, MutVisitor};
use rustc_ast::ptr::P;
use rustc_ast::{self as ast, AstLike};
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
    annotatable: Annotatable,
) -> Vec<Annotatable> {
    check_builtin_macro_attribute(ecx, meta_item, sym::cfg_eval);
    cfg_eval(ecx, annotatable)
}

crate fn cfg_eval(ecx: &ExtCtxt<'_>, annotatable: Annotatable) -> Vec<Annotatable> {
    let mut visitor = CfgEval {
        cfg: StripUnconfigured { sess: ecx.sess, features: ecx.ecfg.features, modified: false },
    };
    let mut annotatable = visitor.configure_annotatable(annotatable);
    if visitor.cfg.modified {
        // Erase the tokens if cfg-stripping modified the item
        // This will cause us to synthesize fake tokens
        // when `nt_to_tokenstream` is called on this item.
        if let Some(tokens) = annotatable.tokens_mut() {
            *tokens = None;
        }
    }
    vec![annotatable]
}

struct CfgEval<'a> {
    cfg: StripUnconfigured<'a>,
}

impl CfgEval<'_> {
    fn configure<T: AstLike>(&mut self, node: T) -> Option<T> {
        self.cfg.configure(node)
    }

    fn configure_annotatable(&mut self, annotatable: Annotatable) -> Annotatable {
        // Since the item itself has already been configured by the InvocationCollector,
        // we know that fold result vector will contain exactly one element
        match annotatable {
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
            Annotatable::ExprField(field) => {
                Annotatable::ExprField(self.flat_map_expr_field(field).pop().unwrap())
            }
            Annotatable::PatField(fp) => {
                Annotatable::PatField(self.flat_map_pat_field(fp).pop().unwrap())
            }
            Annotatable::GenericParam(param) => {
                Annotatable::GenericParam(self.flat_map_generic_param(param).pop().unwrap())
            }
            Annotatable::Param(param) => {
                Annotatable::Param(self.flat_map_param(param).pop().unwrap())
            }
            Annotatable::FieldDef(sf) => {
                Annotatable::FieldDef(self.flat_map_field_def(sf).pop().unwrap())
            }
            Annotatable::Variant(v) => {
                Annotatable::Variant(self.flat_map_variant(v).pop().unwrap())
            }
        }
    }
}

impl MutVisitor for CfgEval<'_> {
    fn visit_expr(&mut self, expr: &mut P<ast::Expr>) {
        self.cfg.configure_expr(expr);
        mut_visit::noop_visit_expr(expr, self);
    }

    fn filter_map_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        let mut expr = configure!(self, expr);
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

    fn flat_map_foreign_item(
        &mut self,
        foreign_item: P<ast::ForeignItem>,
    ) -> SmallVec<[P<ast::ForeignItem>; 1]> {
        mut_visit::noop_flat_map_foreign_item(configure!(self, foreign_item), self)
    }

    fn flat_map_arm(&mut self, arm: ast::Arm) -> SmallVec<[ast::Arm; 1]> {
        mut_visit::noop_flat_map_arm(configure!(self, arm), self)
    }

    fn flat_map_expr_field(&mut self, field: ast::ExprField) -> SmallVec<[ast::ExprField; 1]> {
        mut_visit::noop_flat_map_expr_field(configure!(self, field), self)
    }

    fn flat_map_pat_field(&mut self, fp: ast::PatField) -> SmallVec<[ast::PatField; 1]> {
        mut_visit::noop_flat_map_pat_field(configure!(self, fp), self)
    }

    fn flat_map_param(&mut self, p: ast::Param) -> SmallVec<[ast::Param; 1]> {
        mut_visit::noop_flat_map_param(configure!(self, p), self)
    }

    fn flat_map_field_def(&mut self, sf: ast::FieldDef) -> SmallVec<[ast::FieldDef; 1]> {
        mut_visit::noop_flat_map_field_def(configure!(self, sf), self)
    }

    fn flat_map_variant(&mut self, variant: ast::Variant) -> SmallVec<[ast::Variant; 1]> {
        mut_visit::noop_flat_map_variant(configure!(self, variant), self)
    }
}
