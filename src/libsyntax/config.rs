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
use visit;
use codemap::{Spanned, respan};
use ptr::P;

use util::small_vector::SmallVector;

pub trait CfgFolder: fold::Folder {
    fn configure<T: HasAttrs>(&mut self, node: T) -> Option<T>;
    fn visit_unconfigurable_expr(&mut self, _expr: &ast::Expr) {}
}

/// A folder that strips out items that do not belong in the current
/// configuration.
pub struct StripUnconfigured<'a> {
    diag: CfgDiagReal<'a, 'a>,
    config: &'a ast::CrateConfig,
}

impl<'a> CfgFolder for StripUnconfigured<'a> {
    fn configure<T: HasAttrs>(&mut self, node: T) -> Option<T> {
        if in_cfg(self.config, node.attrs(), &mut self.diag) {
            Some(node)
        } else {
            None
        }
    }

    fn visit_unconfigurable_expr(&mut self, expr: &ast::Expr) {
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
    // Need to do this check here because cfg runs before feature_gates
    check_for_gated_stmt_expr_attributes(&krate, feature_gated_cfgs);

    let krate = process_cfg_attr(diagnostic, krate, feature_gated_cfgs);

    StripUnconfigured {
        config: &krate.config.clone(),
        diag: CfgDiagReal {
            diag: diagnostic,
            feature_gated_cfgs: feature_gated_cfgs,
        },
    }.fold_crate(krate)
}

impl<T: CfgFolder> fold::Folder for T {
    fn fold_foreign_mod(&mut self, foreign_mod: ast::ForeignMod) -> ast::ForeignMod {
        ast::ForeignMod {
            abi: foreign_mod.abi,
            items: foreign_mod.items.into_iter().filter_map(|item| self.configure(item)).collect(),
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
        // If an expr is valid to cfg away it will have been removed by the
        // outer stmt or expression folder before descending in here.
        // Anything else is always required, and thus has to error out
        // in case of a cfg attr.
        //
        // NB: This is intentionally not part of the fold_expr() function
        //     in order for fold_opt_expr() to be able to avoid this check
        self.visit_unconfigurable_expr(&expr);
        fold_expr(self, expr)
    }

    fn fold_opt_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        self.configure(expr).map(|expr| fold_expr(self, expr))
    }

    fn fold_stmt(&mut self, stmt: ast::Stmt) -> SmallVector<ast::Stmt> {
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

// Determine if an item should be translated in the current crate
// configuration based on the item's attributes
fn in_cfg<T: CfgDiag>(cfg: &[P<ast::MetaItem>],
                      attrs: &[ast::Attribute],
                      diag: &mut T) -> bool {
    attrs.iter().all(|attr| {
        let mis = match attr.node.value.node {
            ast::MetaItemKind::List(_, ref mis) if is_cfg(&attr) => mis,
            _ => return true
        };

        if mis.len() != 1 {
            diag.emit_error(|diagnostic| {
                diagnostic.span_err(attr.span, "expected 1 cfg-pattern");
            });
            return true;
        }

        attr::cfg_matches(cfg, &mis[0], diag)
    })
}

struct CfgAttrFolder<'a, T> {
    diag: T,
    config: &'a ast::CrateConfig,
}

// Process `#[cfg_attr]`.
fn process_cfg_attr(diagnostic: &Handler, krate: ast::Crate,
                    feature_gated_cfgs: &mut Vec<GatedCfgAttr>) -> ast::Crate {
    let mut fld = CfgAttrFolder {
        diag: CfgDiagReal {
            diag: diagnostic,
            feature_gated_cfgs: feature_gated_cfgs,
        },
        config: &krate.config.clone(),
    };
    fld.fold_crate(krate)
}

impl<'a, T: CfgDiag> fold::Folder for CfgAttrFolder<'a, T> {
    fn fold_attribute(&mut self, attr: ast::Attribute) -> Option<ast::Attribute> {
        if !attr.check_name("cfg_attr") {
            return fold::noop_fold_attribute(attr, self);
        }

        let attr_list = match attr.meta_item_list() {
            Some(attr_list) => attr_list,
            None => {
                self.diag.emit_error(|diag| {
                    diag.span_err(attr.span,
                        "expected `#[cfg_attr(<cfg pattern>, <attr>)]`");
                });
                return None;
            }
        };
        let (cfg, mi) = match (attr_list.len(), attr_list.get(0), attr_list.get(1)) {
            (2, Some(cfg), Some(mi)) => (cfg, mi),
            _ => {
                self.diag.emit_error(|diag| {
                    diag.span_err(attr.span,
                        "expected `#[cfg_attr(<cfg pattern>, <attr>)]`");
                });
                return None;
            }
        };

        if attr::cfg_matches(&self.config[..], &cfg, &mut self.diag) {
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

    // Need the ability to run pre-expansion.
    fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac {
        fold::noop_fold_mac(mac, self)
    }
}

fn check_for_gated_stmt_expr_attributes(krate: &ast::Crate,
                                        discovered: &mut Vec<GatedCfgAttr>) {
    let mut v = StmtExprAttrFeatureVisitor {
        config: &krate.config,
        discovered: discovered,
    };
    visit::walk_crate(&mut v, krate);
}

/// To cover this feature, we need to discover all attributes
/// so we need to run before cfg.
struct StmtExprAttrFeatureVisitor<'a, 'b> {
    config: &'a ast::CrateConfig,
    discovered: &'b mut Vec<GatedCfgAttr>,
}

// Runs the cfg_attr and cfg folders locally in "silent" mode
// to discover attribute use on stmts or expressions ahead of time
impl<'v, 'a, 'b> visit::Visitor<'v> for StmtExprAttrFeatureVisitor<'a, 'b> {
    fn visit_stmt(&mut self, s: &'v ast::Stmt) {
        // check if there even are any attributes on this node
        let stmt_attrs = s.node.attrs();
        if stmt_attrs.len() > 0 {
            // attributes on items are fine
            if let ast::StmtKind::Decl(ref decl, _) = s.node {
                if let ast::DeclKind::Item(_) = decl.node {
                    visit::walk_stmt(self, s);
                    return;
                }
            }

            // flag the offending attributes
            for attr in stmt_attrs {
                self.discovered.push(GatedCfgAttr::GatedAttr(attr.span));
            }

            // if the node does not end up being cfg-d away, walk down
            if node_survives_cfg(stmt_attrs, self.config) {
                visit::walk_stmt(self, s);
            }
        } else {
            visit::walk_stmt(self, s);
        }
    }

    fn visit_expr(&mut self, ex: &'v ast::Expr) {
        // check if there even are any attributes on this node
        let expr_attrs = ex.attrs();
        if expr_attrs.len() > 0 {

            // flag the offending attributes
            for attr in expr_attrs {
                self.discovered.push(GatedCfgAttr::GatedAttr(attr.span));
            }

            // if the node does not end up being cfg-d away, walk down
            if node_survives_cfg(expr_attrs, self.config) {
                visit::walk_expr(self, ex);
            }
        } else {
            visit::walk_expr(self, ex);
        }
    }

    fn visit_foreign_item(&mut self, i: &'v ast::ForeignItem) {
        if node_survives_cfg(&i.attrs, self.config) {
            visit::walk_foreign_item(self, i);
        }
    }

    fn visit_item(&mut self, i: &'v ast::Item) {
        if node_survives_cfg(&i.attrs, self.config) {
            visit::walk_item(self, i);
        }
    }

    fn visit_impl_item(&mut self, ii: &'v ast::ImplItem) {
        if node_survives_cfg(&ii.attrs, self.config) {
            visit::walk_impl_item(self, ii);
        }
    }

    fn visit_trait_item(&mut self, ti: &'v ast::TraitItem) {
        if node_survives_cfg(&ti.attrs, self.config) {
            visit::walk_trait_item(self, ti);
        }
    }

    fn visit_struct_field(&mut self, s: &'v ast::StructField) {
        if node_survives_cfg(&s.attrs, self.config) {
            visit::walk_struct_field(self, s);
        }
    }

    fn visit_variant(&mut self, v: &'v ast::Variant,
                     g: &'v ast::Generics, item_id: ast::NodeId) {
        if node_survives_cfg(&v.node.attrs, self.config) {
            visit::walk_variant(self, v, g, item_id);
        }
    }

    fn visit_arm(&mut self, a: &'v ast::Arm) {
        if node_survives_cfg(&a.attrs, self.config) {
            visit::walk_arm(self, a);
        }
    }

    // This visitor runs pre expansion, so we need to prevent
    // the default panic here
    fn visit_mac(&mut self, mac: &'v ast::Mac) {
        visit::walk_mac(self, mac)
    }
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

struct CfgDiagSilent {
    error: bool,
}

impl CfgDiag for CfgDiagSilent {
    fn emit_error<F>(&mut self, _: F) where F: FnMut(&Handler) {
        self.error = true;
    }
    fn flag_gated<F>(&mut self, _: F) where F: FnMut(&mut Vec<GatedCfgAttr>) {}
}

fn node_survives_cfg(attrs: &[ast::Attribute],
                     config: &ast::CrateConfig) -> bool {
    let mut survives_cfg = true;

    for attr in attrs {
        let mut fld = CfgAttrFolder {
            diag: CfgDiagSilent { error: false },
            config: config,
        };
        let attr = fld.fold_attribute(attr.clone());

        // In case of error we can just return true,
        // since the actual cfg folders will end compilation anyway.

        if fld.diag.error { return true; }

        survives_cfg &= attr.map(|attr| {
            let mut diag = CfgDiagSilent { error: false };
            let r = in_cfg(config, &[attr], &mut diag);
            if diag.error { return true; }
            r
        }).unwrap_or(true)
    }

    survives_cfg
}
