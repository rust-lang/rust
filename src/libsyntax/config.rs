// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use attr::AttrMetaMethods;
use diagnostic::SpanHandler;
use feature_gate::GatedCfg;
use fold::Folder;
use {ast, fold, attr};
use codemap::{Spanned, respan};
use ptr::P;

use util::small_vector::SmallVector;

/// A folder that strips out items that do not belong in the current
/// configuration.
struct Context<'a, F> where F: FnMut(&[ast::Attribute]) -> bool {
    in_cfg: F,
    diagnostic: &'a SpanHandler,
}

// Support conditional compilation by transforming the AST, stripping out
// any items that do not belong in the current configuration
pub fn strip_unconfigured_items(diagnostic: &SpanHandler, krate: ast::Crate,
                                feature_gated_cfgs: &mut Vec<GatedCfg>)
                                -> ast::Crate
{
    let krate = process_cfg_attr(diagnostic, krate, feature_gated_cfgs);
    let config = krate.config.clone();
    strip_items(diagnostic,
                krate,
                |attrs| in_cfg(diagnostic, &config, attrs, feature_gated_cfgs))
}

impl<'a, F> fold::Folder for Context<'a, F> where F: FnMut(&[ast::Attribute]) -> bool {
    fn fold_mod(&mut self, module: ast::Mod) -> ast::Mod {
        fold_mod(self, module)
    }
    fn fold_foreign_mod(&mut self, foreign_mod: ast::ForeignMod) -> ast::ForeignMod {
        fold_foreign_mod(self, foreign_mod)
    }
    fn fold_item_underscore(&mut self, item: ast::Item_) -> ast::Item_ {
        fold_item_underscore(self, item)
    }
    fn fold_expr(&mut self, expr: P<ast::Expr>) -> P<ast::Expr> {
        // If an expr is valid to cfg away it will have been removed by the
        // outer stmt or expression folder before descending in here.
        // Anything else is always required, and thus has to error out
        // in case of a cfg attr.
        //
        // NB: This is intentionally not part of the fold_expr() function
        //     in order for fold_opt_expr() to be able to avoid this check
        if let Some(attr) = expr.attrs().iter().find(|a| is_cfg(a)) {
            self.diagnostic.span_err(attr.span,
                "removing an expression is not supported in this position");
        }
        fold_expr(self, expr)
    }
    fn fold_opt_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        fold_opt_expr(self, expr)
    }
    fn fold_stmt(&mut self, stmt: P<ast::Stmt>) -> SmallVector<P<ast::Stmt>> {
        fold_stmt(self, stmt)
    }
    fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac {
        fold::noop_fold_mac(mac, self)
    }
    fn fold_item(&mut self, item: P<ast::Item>) -> SmallVector<P<ast::Item>> {
        fold_item(self, item)
    }
}

pub fn strip_items<'a, F>(diagnostic: &'a SpanHandler,
                          krate: ast::Crate, in_cfg: F) -> ast::Crate where
    F: FnMut(&[ast::Attribute]) -> bool,
{
    let mut ctxt = Context {
        in_cfg: in_cfg,
        diagnostic: diagnostic,
    };
    ctxt.fold_crate(krate)
}

fn fold_mod<F>(cx: &mut Context<F>,
               ast::Mod {inner, items}: ast::Mod)
               -> ast::Mod where
    F: FnMut(&[ast::Attribute]) -> bool
{
    ast::Mod {
        inner: inner,
        items: items.into_iter().flat_map(|a| {
            cx.fold_item(a).into_iter()
        }).collect()
    }
}

fn filter_foreign_item<F>(cx: &mut Context<F>,
                          item: P<ast::ForeignItem>)
                          -> Option<P<ast::ForeignItem>> where
    F: FnMut(&[ast::Attribute]) -> bool
{
    if foreign_item_in_cfg(cx, &item) {
        Some(item)
    } else {
        None
    }
}

fn fold_foreign_mod<F>(cx: &mut Context<F>,
                       ast::ForeignMod {abi, items}: ast::ForeignMod)
                       -> ast::ForeignMod where
    F: FnMut(&[ast::Attribute]) -> bool
{
    ast::ForeignMod {
        abi: abi,
        items: items.into_iter()
                    .filter_map(|a| filter_foreign_item(cx, a))
                    .collect()
    }
}

fn fold_item<F>(cx: &mut Context<F>, item: P<ast::Item>) -> SmallVector<P<ast::Item>> where
    F: FnMut(&[ast::Attribute]) -> bool
{
    if item_in_cfg(cx, &item) {
        SmallVector::one(item.map(|i| cx.fold_item_simple(i)))
    } else {
        SmallVector::zero()
    }
}

fn fold_item_underscore<F>(cx: &mut Context<F>, item: ast::Item_) -> ast::Item_ where
    F: FnMut(&[ast::Attribute]) -> bool
{
    let item = match item {
        ast::ItemImpl(u, o, a, b, c, impl_items) => {
            let impl_items = impl_items.into_iter()
                                       .filter(|ii| (cx.in_cfg)(&ii.attrs))
                                       .collect();
            ast::ItemImpl(u, o, a, b, c, impl_items)
        }
        ast::ItemTrait(u, a, b, methods) => {
            let methods = methods.into_iter()
                                 .filter(|ti| (cx.in_cfg)(&ti.attrs))
                                 .collect();
            ast::ItemTrait(u, a, b, methods)
        }
        ast::ItemStruct(def, generics) => {
            ast::ItemStruct(fold_struct(cx, def), generics)
        }
        ast::ItemEnum(def, generics) => {
            let variants = def.variants.into_iter().filter_map(|v| {
                if !(cx.in_cfg)(&v.node.attrs) {
                    None
                } else {
                    Some(v.map(|Spanned {node: ast::Variant_ {name, attrs, data,
                                                              disr_expr}, span}| {
                        Spanned {
                            node: ast::Variant_ {
                                name: name,
                                attrs: attrs,
                                data: fold_struct(cx, data),
                                disr_expr: disr_expr,
                            },
                            span: span
                        }
                    }))
                }
            });
            ast::ItemEnum(ast::EnumDef {
                variants: variants.collect(),
            }, generics)
        }
        item => item,
    };

    fold::noop_fold_item_underscore(item, cx)
}

fn fold_struct<F>(cx: &mut Context<F>, vdata: ast::VariantData) -> ast::VariantData where
    F: FnMut(&[ast::Attribute]) -> bool
{
    match vdata {
        ast::VariantData::Struct(fields, id) => {
            ast::VariantData::Struct(fields.into_iter().filter(|m| {
                (cx.in_cfg)(&m.node.attrs)
            }).collect(), id)
        }
        ast::VariantData::Tuple(fields, id) => {
            ast::VariantData::Tuple(fields.into_iter().filter(|m| {
                (cx.in_cfg)(&m.node.attrs)
            }).collect(), id)
        }
        ast::VariantData::Unit(id) => ast::VariantData::Unit(id)
    }
}

fn fold_opt_expr<F>(cx: &mut Context<F>, expr: P<ast::Expr>) -> Option<P<ast::Expr>>
    where F: FnMut(&[ast::Attribute]) -> bool
{
    if expr_in_cfg(cx, &expr) {
        Some(fold_expr(cx, expr))
    } else {
        None
    }
}

fn fold_expr<F>(cx: &mut Context<F>, expr: P<ast::Expr>) -> P<ast::Expr> where
    F: FnMut(&[ast::Attribute]) -> bool
{
    expr.map(|ast::Expr {id, span, node, attrs}| {
        fold::noop_fold_expr(ast::Expr {
            id: id,
            node: match node {
                ast::ExprMatch(m, arms) => {
                    ast::ExprMatch(m, arms.into_iter()
                                        .filter(|a| (cx.in_cfg)(&a.attrs))
                                        .collect())
                }
                _ => node
            },
            span: span,
            attrs: attrs,
        }, cx)
    })
}

fn fold_stmt<F>(cx: &mut Context<F>, stmt: P<ast::Stmt>) -> SmallVector<P<ast::Stmt>>
    where F: FnMut(&[ast::Attribute]) -> bool
{
    if stmt_in_cfg(cx, &stmt) {
        stmt.and_then(|s| fold::noop_fold_stmt(s, cx))
    } else {
        SmallVector::zero()
    }
}

fn stmt_in_cfg<F>(cx: &mut Context<F>, stmt: &ast::Stmt) -> bool where
    F: FnMut(&[ast::Attribute]) -> bool
{
    (cx.in_cfg)(stmt.node.attrs())
}

fn expr_in_cfg<F>(cx: &mut Context<F>, expr: &ast::Expr) -> bool where
    F: FnMut(&[ast::Attribute]) -> bool
{
    (cx.in_cfg)(expr.attrs())
}

fn item_in_cfg<F>(cx: &mut Context<F>, item: &ast::Item) -> bool where
    F: FnMut(&[ast::Attribute]) -> bool
{
    return (cx.in_cfg)(&item.attrs);
}

fn foreign_item_in_cfg<F>(cx: &mut Context<F>, item: &ast::ForeignItem) -> bool where
    F: FnMut(&[ast::Attribute]) -> bool
{
    return (cx.in_cfg)(&item.attrs);
}

fn is_cfg(attr: &ast::Attribute) -> bool {
    attr.check_name("cfg")
}

// Determine if an item should be translated in the current crate
// configuration based on the item's attributes
fn in_cfg(diagnostic: &SpanHandler,
          cfg: &[P<ast::MetaItem>],
          attrs: &[ast::Attribute],
          feature_gated_cfgs: &mut Vec<GatedCfg>) -> bool {
    attrs.iter().all(|attr| {
        let mis = match attr.node.value.node {
            ast::MetaList(_, ref mis) if is_cfg(&attr) => mis,
            _ => return true
        };

        if mis.len() != 1 {
            diagnostic.span_err(attr.span, "expected 1 cfg-pattern");
            return true;
        }

        attr::cfg_matches(diagnostic, cfg, &mis[0],
                          feature_gated_cfgs)
    })
}

struct CfgAttrFolder<'a, 'b> {
    diag: &'a SpanHandler,
    config: ast::CrateConfig,
    feature_gated_cfgs: &'b mut Vec<GatedCfg>
}

// Process `#[cfg_attr]`.
fn process_cfg_attr(diagnostic: &SpanHandler, krate: ast::Crate,
                    feature_gated_cfgs: &mut Vec<GatedCfg>) -> ast::Crate {
    let mut fld = CfgAttrFolder {
        diag: diagnostic,
        config: krate.config.clone(),
        feature_gated_cfgs: feature_gated_cfgs,
    };
    fld.fold_crate(krate)
}

impl<'a,'b> fold::Folder for CfgAttrFolder<'a,'b> {
    fn fold_attribute(&mut self, attr: ast::Attribute) -> Option<ast::Attribute> {
        if !attr.check_name("cfg_attr") {
            return fold::noop_fold_attribute(attr, self);
        }

        let attr_list = match attr.meta_item_list() {
            Some(attr_list) => attr_list,
            None => {
                self.diag.span_err(attr.span, "expected `#[cfg_attr(<cfg pattern>, <attr>)]`");
                return None;
            }
        };
        let (cfg, mi) = match (attr_list.len(), attr_list.get(0), attr_list.get(1)) {
            (2, Some(cfg), Some(mi)) => (cfg, mi),
            _ => {
                self.diag.span_err(attr.span, "expected `#[cfg_attr(<cfg pattern>, <attr>)]`");
                return None;
            }
        };

        if attr::cfg_matches(self.diag, &self.config[..], &cfg,
                             self.feature_gated_cfgs) {
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
