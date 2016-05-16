// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use attr::{AttrMetaMethods, HasAttrs};
use errors::Handler;
use feature_gate::GatedCfgAttr;
use fold::Folder;
use {ast, fold, attr};
use codemap::{Spanned, respan};
use ptr::P;

use util::small_vector::SmallVector;

pub trait CfgFolder: fold::Folder {
    fn in_cfg(&mut self, attrs: &[ast::Attribute]) -> bool;
    fn process_attrs<T: HasAttrs>(&mut self, node: T) -> T { node }
    fn visit_stmt_or_expr_attrs(&mut self, _attrs: &[ast::Attribute]) {}
    fn visit_unremovable_expr(&mut self, _expr: &ast::Expr) {}

    fn configure<T: HasAttrs>(&mut self, node: T) -> Option<T> {
        let node = self.process_attrs(node);
        if self.in_cfg(node.attrs()) { Some(node) } else { None }
    }
}

/// A folder that strips out items that do not belong in the current
/// configuration.
pub struct StripUnconfigured<'a> {
    diag: CfgDiagReal<'a, 'a>,
    config: &'a ast::CrateConfig,
}

impl<'a> StripUnconfigured<'a> {
    pub fn new(config: &'a ast::CrateConfig,
               diagnostic: &'a Handler,
               feature_gated_cfgs: &'a mut Vec<GatedCfgAttr>)
               -> Self {
        StripUnconfigured {
            config: config,
            diag: CfgDiagReal { diag: diagnostic, feature_gated_cfgs: feature_gated_cfgs },
        }
    }

    fn process_cfg_attr(&mut self, attr: ast::Attribute) -> Option<ast::Attribute> {
        if !attr.check_name("cfg_attr") {
            return Some(attr);
        }

        let attr_list = match attr.meta_item_list() {
            Some(attr_list) => attr_list,
            None => {
                let msg = "expected `#[cfg_attr(<cfg pattern>, <attr>)]`";
                self.diag.diag.span_err(attr.span, msg);
                return None;
            }
        };
        let (cfg, mi) = match (attr_list.len(), attr_list.get(0), attr_list.get(1)) {
            (2, Some(cfg), Some(mi)) => (cfg, mi),
            _ => {
                let msg = "expected `#[cfg_attr(<cfg pattern>, <attr>)]`";
                self.diag.diag.span_err(attr.span, msg);
                return None;
            }
        };

        if attr::cfg_matches(self.config, &cfg, &mut self.diag) {
            Some(respan(mi.span, ast::Attribute_ {
                id: attr::mk_attr_id(),
                style: attr.node.style,
                value: mi.clone(),
                is_sugared_doc: false,
            }))
        } else {
            None
        }
    }
}

impl<'a> CfgFolder for StripUnconfigured<'a> {
    // Determine if an item should be translated in the current crate
    // configuration based on the item's attributes
    fn in_cfg(&mut self, attrs: &[ast::Attribute]) -> bool {
        attrs.iter().all(|attr| {
            let mis = match attr.node.value.node {
                ast::MetaItemKind::List(_, ref mis) if is_cfg(&attr) => mis,
                _ => return true
            };

            if mis.len() != 1 {
                self.diag.emit_error(|diagnostic| {
                    diagnostic.span_err(attr.span, "expected 1 cfg-pattern");
                });
                return true;
            }

            attr::cfg_matches(self.config, &mis[0], &mut self.diag)
        })
    }

    fn process_attrs<T: HasAttrs>(&mut self, node: T) -> T {
        node.map_attrs(|attrs| {
            attrs.into_iter().filter_map(|attr| self.process_cfg_attr(attr)).collect()
        })
    }

    fn visit_stmt_or_expr_attrs(&mut self, attrs: &[ast::Attribute]) {
        // flag the offending attributes
        for attr in attrs.iter() {
            self.diag.feature_gated_cfgs.push(GatedCfgAttr::GatedAttr(attr.span));
        }
    }

    fn visit_unremovable_expr(&mut self, expr: &ast::Expr) {
        if let Some(attr) = expr.attrs().iter().find(|a| is_cfg(a)) {
            let msg = "removing an expression is not supported in this position";
            self.diag.diag.span_err(attr.span, msg);
        }
    }
}

// Support conditional compilation by transforming the AST, stripping out
// any items that do not belong in the current configuration
pub fn strip_unconfigured_items(diagnostic: &Handler, krate: ast::Crate,
                                feature_gated_cfgs: &mut Vec<GatedCfgAttr>)
                                -> ast::Crate
{
    let config = &krate.config.clone();
    StripUnconfigured::new(config, diagnostic, feature_gated_cfgs).fold_crate(krate)
}

impl<T: CfgFolder> fold::Folder for T {
    fn fold_foreign_mod(&mut self, foreign_mod: ast::ForeignMod) -> ast::ForeignMod {
        ast::ForeignMod {
            abi: foreign_mod.abi,
            items: foreign_mod.items.into_iter().filter_map(|item| {
                self.configure(item).map(|item| fold::noop_fold_foreign_item(item, self))
            }).collect(),
        }
    }

    fn fold_item_kind(&mut self, item: ast::ItemKind) -> ast::ItemKind {
        let fold_struct = |this: &mut Self, vdata| match vdata {
            ast::VariantData::Struct(fields, id) => {
                let fields = fields.into_iter().filter_map(|field| this.configure(field));
                ast::VariantData::Struct(fields.collect(), id)
            }
            ast::VariantData::Tuple(fields, id) => {
                let fields = fields.into_iter().filter_map(|field| this.configure(field));
                ast::VariantData::Tuple(fields.collect(), id)
            }
            ast::VariantData::Unit(id) => ast::VariantData::Unit(id)
        };

        let item = match item {
            ast::ItemKind::Impl(u, o, a, b, c, items) => {
                let items = items.into_iter().filter_map(|item| self.configure(item)).collect();
                ast::ItemKind::Impl(u, o, a, b, c, items)
            }
            ast::ItemKind::Trait(u, a, b, items) => {
                let items = items.into_iter().filter_map(|item| self.configure(item)).collect();
                ast::ItemKind::Trait(u, a, b, items)
            }
            ast::ItemKind::Struct(def, generics) => {
                ast::ItemKind::Struct(fold_struct(self, def), generics)
            }
            ast::ItemKind::Enum(def, generics) => {
                let variants = def.variants.into_iter().filter_map(|v| {
                    self.configure(v).map(|v| {
                        Spanned {
                            node: ast::Variant_ {
                                name: v.node.name,
                                attrs: v.node.attrs,
                                data: fold_struct(self, v.node.data),
                                disr_expr: v.node.disr_expr,
                            },
                            span: v.span
                        }
                    })
                });
                ast::ItemKind::Enum(ast::EnumDef {
                    variants: variants.collect(),
                }, generics)
            }
            item => item,
        };

        fold::noop_fold_item_kind(item, self)
    }

    fn fold_expr(&mut self, expr: P<ast::Expr>) -> P<ast::Expr> {
        self.visit_stmt_or_expr_attrs(expr.attrs());
        // If an expr is valid to cfg away it will have been removed by the
        // outer stmt or expression folder before descending in here.
        // Anything else is always required, and thus has to error out
        // in case of a cfg attr.
        //
        // NB: This is intentionally not part of the fold_expr() function
        //     in order for fold_opt_expr() to be able to avoid this check
        self.visit_unremovable_expr(&expr);
        let expr = self.process_attrs(expr);
        fold_expr(self, expr)
    }

    fn fold_opt_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        self.configure(expr).map(|expr| fold_expr(self, expr))
    }

    fn fold_stmt(&mut self, stmt: ast::Stmt) -> SmallVector<ast::Stmt> {
        let is_item = match stmt.node {
            ast::StmtKind::Decl(ref decl, _) => match decl.node {
                ast::DeclKind::Item(_) => true,
                _ => false,
            },
            _ => false,
        };

        // avoid calling `visit_stmt_or_expr_attrs` on items
        if !is_item {
            self.visit_stmt_or_expr_attrs(stmt.attrs());
        }

        self.configure(stmt).map(|stmt| fold::noop_fold_stmt(stmt, self))
                            .unwrap_or(SmallVector::zero())
    }

    fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac {
        fold::noop_fold_mac(mac, self)
    }

    fn fold_item(&mut self, item: P<ast::Item>) -> SmallVector<P<ast::Item>> {
        self.configure(item).map(|item| SmallVector::one(item.map(|i| self.fold_item_simple(i))))
                            .unwrap_or(SmallVector::zero())
    }
}

fn fold_expr<F: CfgFolder>(folder: &mut F, expr: P<ast::Expr>) -> P<ast::Expr> {
    expr.map(|ast::Expr {id, span, node, attrs}| {
        fold::noop_fold_expr(ast::Expr {
            id: id,
            node: match node {
                ast::ExprKind::Match(m, arms) => {
                    ast::ExprKind::Match(m, arms.into_iter()
                                        .filter_map(|a| folder.configure(a))
                                        .collect())
                }
                _ => node
            },
            span: span,
            attrs: attrs,
        }, folder)
    })
}

fn is_cfg(attr: &ast::Attribute) -> bool {
    attr.check_name("cfg")
}

pub trait CfgDiag {
    fn emit_error<F>(&mut self, f: F) where F: FnMut(&Handler);
    fn flag_gated<F>(&mut self, f: F) where F: FnMut(&mut Vec<GatedCfgAttr>);
}

pub struct CfgDiagReal<'a, 'b> {
    pub diag: &'a Handler,
    pub feature_gated_cfgs: &'b mut Vec<GatedCfgAttr>,
}

impl<'a, 'b> CfgDiag for CfgDiagReal<'a, 'b> {
    fn emit_error<F>(&mut self, mut f: F) where F: FnMut(&Handler) {
        f(self.diag)
    }
    fn flag_gated<F>(&mut self, mut f: F) where F: FnMut(&mut Vec<GatedCfgAttr>) {
        f(self.feature_gated_cfgs)
    }
}
