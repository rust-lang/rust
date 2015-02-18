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
use fold::Folder;
use {ast, fold, attr};
use codemap::{Spanned, respan};
use ptr::P;

use util::small_vector::SmallVector;

/// A folder that strips out items that do not belong in the current
/// configuration.
struct Context<F> where F: FnMut(&[ast::Attribute]) -> bool {
    in_cfg: F,
}

// Support conditional compilation by transforming the AST, stripping out
// any items that do not belong in the current configuration
pub fn strip_unconfigured_items(diagnostic: &SpanHandler, krate: ast::Crate) -> ast::Crate {
    let krate = process_cfg_attr(diagnostic, krate);
    let config = krate.config.clone();
    strip_items(krate, |attrs| in_cfg(diagnostic, &config, attrs))
}

impl<F> fold::Folder for Context<F> where F: FnMut(&[ast::Attribute]) -> bool {
    fn fold_mod(&mut self, module: ast::Mod) -> ast::Mod {
        fold_mod(self, module)
    }
    fn fold_block(&mut self, block: P<ast::Block>) -> P<ast::Block> {
        fold_block(self, block)
    }
    fn fold_foreign_mod(&mut self, foreign_mod: ast::ForeignMod) -> ast::ForeignMod {
        fold_foreign_mod(self, foreign_mod)
    }
    fn fold_item_underscore(&mut self, item: ast::Item_) -> ast::Item_ {
        fold_item_underscore(self, item)
    }
    fn fold_expr(&mut self, expr: P<ast::Expr>) -> P<ast::Expr> {
        fold_expr(self, expr)
    }
    fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac {
        fold::noop_fold_mac(mac, self)
    }
    fn fold_item(&mut self, item: P<ast::Item>) -> SmallVector<P<ast::Item>> {
        fold_item(self, item)
    }
}

pub fn strip_items<F>(krate: ast::Crate, in_cfg: F) -> ast::Crate where
    F: FnMut(&[ast::Attribute]) -> bool,
{
    let mut ctxt = Context {
        in_cfg: in_cfg,
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
    if foreign_item_in_cfg(cx, &*item) {
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
    if item_in_cfg(cx, &*item) {
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
                                       .filter(|ii| impl_item_in_cfg(cx, ii))
                                       .collect();
            ast::ItemImpl(u, o, a, b, c, impl_items)
        }
        ast::ItemTrait(u, a, b, methods) => {
            let methods = methods.into_iter()
                                 .filter(|m| trait_method_in_cfg(cx, m))
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
                    Some(v.map(|Spanned {node: ast::Variant_ {id, name, attrs, kind,
                                                              disr_expr, vis}, span}| {
                        Spanned {
                            node: ast::Variant_ {
                                id: id,
                                name: name,
                                attrs: attrs,
                                kind: match kind {
                                    ast::TupleVariantKind(..) => kind,
                                    ast::StructVariantKind(def) => {
                                        ast::StructVariantKind(fold_struct(cx, def))
                                    }
                                },
                                disr_expr: disr_expr,
                                vis: vis
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

fn fold_struct<F>(cx: &mut Context<F>, def: P<ast::StructDef>) -> P<ast::StructDef> where
    F: FnMut(&[ast::Attribute]) -> bool
{
    def.map(|ast::StructDef { fields, ctor_id }| {
        ast::StructDef {
            fields: fields.into_iter().filter(|m| {
                (cx.in_cfg)(&m.node.attrs)
            }).collect(),
            ctor_id: ctor_id,
        }
    })
}

fn retain_stmt<F>(cx: &mut Context<F>, stmt: &ast::Stmt) -> bool where
    F: FnMut(&[ast::Attribute]) -> bool
{
    match stmt.node {
        ast::StmtDecl(ref decl, _) => {
            match decl.node {
                ast::DeclItem(ref item) => {
                    item_in_cfg(cx, &**item)
                }
                _ => true
            }
        }
        _ => true
    }
}

fn fold_block<F>(cx: &mut Context<F>, b: P<ast::Block>) -> P<ast::Block> where
    F: FnMut(&[ast::Attribute]) -> bool
{
    b.map(|ast::Block {id, stmts, expr, rules, span}| {
        let resulting_stmts: Vec<P<ast::Stmt>> =
            stmts.into_iter().filter(|a| retain_stmt(cx, &**a)).collect();
        let resulting_stmts = resulting_stmts.into_iter()
            .flat_map(|stmt| cx.fold_stmt(stmt).into_iter())
            .collect();
        ast::Block {
            id: id,
            stmts: resulting_stmts,
            expr: expr.map(|x| cx.fold_expr(x)),
            rules: rules,
            span: span,
        }
    })
}

fn fold_expr<F>(cx: &mut Context<F>, expr: P<ast::Expr>) -> P<ast::Expr> where
    F: FnMut(&[ast::Attribute]) -> bool
{
    expr.map(|ast::Expr {id, span, node}| {
        fold::noop_fold_expr(ast::Expr {
            id: id,
            node: match node {
                ast::ExprMatch(m, arms, source) => {
                    ast::ExprMatch(m, arms.into_iter()
                                        .filter(|a| (cx.in_cfg)(&a.attrs))
                                        .collect(), source)
                }
                _ => node
            },
            span: span
        }, cx)
    })
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

fn trait_method_in_cfg<F>(cx: &mut Context<F>, meth: &ast::TraitItem) -> bool where
    F: FnMut(&[ast::Attribute]) -> bool
{
    match *meth {
        ast::RequiredMethod(ref meth) => (cx.in_cfg)(&meth.attrs),
        ast::ProvidedMethod(ref meth) => (cx.in_cfg)(&meth.attrs),
        ast::TypeTraitItem(ref typ) => (cx.in_cfg)(&typ.attrs),
    }
}

fn impl_item_in_cfg<F>(cx: &mut Context<F>, impl_item: &ast::ImplItem) -> bool where
    F: FnMut(&[ast::Attribute]) -> bool
{
    match *impl_item {
        ast::MethodImplItem(ref meth) => (cx.in_cfg)(&meth.attrs),
        ast::TypeImplItem(ref typ) => (cx.in_cfg)(&typ.attrs),
    }
}

// Determine if an item should be translated in the current crate
// configuration based on the item's attributes
fn in_cfg(diagnostic: &SpanHandler, cfg: &[P<ast::MetaItem>], attrs: &[ast::Attribute]) -> bool {
    attrs.iter().all(|attr| {
        let mis = match attr.node.value.node {
            ast::MetaList(_, ref mis) if attr.check_name("cfg") => mis,
            _ => return true
        };

        if mis.len() != 1 {
            diagnostic.span_err(attr.span, "expected 1 cfg-pattern");
            return true;
        }

        attr::cfg_matches(diagnostic, cfg, &*mis[0])
    })
}

struct CfgAttrFolder<'a> {
    diag: &'a SpanHandler,
    config: ast::CrateConfig,
}

// Process `#[cfg_attr]`.
fn process_cfg_attr(diagnostic: &SpanHandler, krate: ast::Crate) -> ast::Crate {
    let mut fld = CfgAttrFolder {
        diag: diagnostic,
        config: krate.config.clone(),
    };
    fld.fold_crate(krate)
}

impl<'a> fold::Folder for CfgAttrFolder<'a> {
    fn fold_attribute(&mut self, attr: ast::Attribute) -> Option<ast::Attribute> {
        if !attr.check_name("cfg_attr") {
            return fold::noop_fold_attribute(attr, self);
        }

        let (cfg, mi) = match attr.meta_item_list() {
            Some([ref cfg, ref mi]) => (cfg, mi),
            _ => {
                self.diag.span_err(attr.span, "expected `#[cfg_attr(<cfg pattern>, <attr>)]`");
                return None;
            }
        };

        if attr::cfg_matches(self.diag, &self.config[..], &cfg) {
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
