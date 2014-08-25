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
use codemap::Spanned;
use ptr::P;

/// A folder that strips out items that do not belong in the current
/// configuration.
struct Context<'a> {
    in_cfg: |attrs: &[ast::Attribute]|: 'a -> bool,
}

// Support conditional compilation by transforming the AST, stripping out
// any items that do not belong in the current configuration
pub fn strip_unconfigured_items(diagnostic: &SpanHandler, krate: ast::Crate) -> ast::Crate {
    let config = krate.config.clone();
    strip_items(krate, |attrs| in_cfg(diagnostic, config.as_slice(), attrs))
}

impl<'a> fold::Folder for Context<'a> {
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
}

pub fn strip_items(krate: ast::Crate,
                   in_cfg: |attrs: &[ast::Attribute]| -> bool)
                   -> ast::Crate {
    let mut ctxt = Context {
        in_cfg: in_cfg,
    };
    ctxt.fold_crate(krate)
}

fn filter_view_item(cx: &mut Context, view_item: ast::ViewItem) -> Option<ast::ViewItem> {
    if view_item_in_cfg(cx, &view_item) {
        Some(view_item)
    } else {
        None
    }
}

fn fold_mod(cx: &mut Context, ast::Mod {inner, view_items, items}: ast::Mod) -> ast::Mod {
    ast::Mod {
        inner: inner,
        view_items: view_items.into_iter().filter_map(|a| {
            filter_view_item(cx, a).map(|x| cx.fold_view_item(x))
        }).collect(),
        items: items.into_iter().filter_map(|a| {
            if item_in_cfg(cx, &*a) {
                Some(cx.fold_item(a))
            } else {
                None
            }
        }).flat_map(|x| x.into_iter()).collect()
    }
}

fn filter_foreign_item(cx: &mut Context, item: P<ast::ForeignItem>)
                       -> Option<P<ast::ForeignItem>> {
    if foreign_item_in_cfg(cx, &*item) {
        Some(item)
    } else {
        None
    }
}

fn fold_foreign_mod(cx: &mut Context, ast::ForeignMod {abi, view_items, items}: ast::ForeignMod)
                    -> ast::ForeignMod {
    ast::ForeignMod {
        abi: abi,
        view_items: view_items.into_iter().filter_map(|a| {
            filter_view_item(cx, a).map(|x| cx.fold_view_item(x))
        }).collect(),
        items: items.into_iter()
                    .filter_map(|a| filter_foreign_item(cx, a))
                    .collect()
    }
}

fn fold_item_underscore(cx: &mut Context, item: ast::Item_) -> ast::Item_ {
    let item = match item {
        ast::ItemImpl(a, b, c, impl_items) => {
            let impl_items = impl_items.into_iter()
                                       .filter(|ii| impl_item_in_cfg(cx, ii))
                                       .collect();
            ast::ItemImpl(a, b, c, impl_items)
        }
        ast::ItemTrait(a, b, c, methods) => {
            let methods = methods.into_iter()
                                 .filter(|m| trait_method_in_cfg(cx, m))
                                 .collect();
            ast::ItemTrait(a, b, c, methods)
        }
        ast::ItemStruct(def, generics) => {
            ast::ItemStruct(fold_struct(cx, def), generics)
        }
        ast::ItemEnum(def, generics) => {
            let mut variants = def.variants.into_iter().filter_map(|v| {
                if !(cx.in_cfg)(v.node.attrs.as_slice()) {
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

fn fold_struct(cx: &mut Context, def: P<ast::StructDef>) -> P<ast::StructDef> {
    def.map(|ast::StructDef {fields, ctor_id, super_struct, is_virtual}| {
        ast::StructDef {
            fields: fields.into_iter().filter(|m| {
                (cx.in_cfg)(m.node.attrs.as_slice())
            }).collect(),
            ctor_id: ctor_id,
            super_struct: super_struct,
            is_virtual: is_virtual,
        }
    })
}

fn retain_stmt(cx: &mut Context, stmt: &ast::Stmt) -> bool {
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

fn fold_block(cx: &mut Context, b: P<ast::Block>) -> P<ast::Block> {
    b.map(|ast::Block {id, view_items, stmts, expr, rules, span}| {
        let resulting_stmts: Vec<P<ast::Stmt>> =
            stmts.into_iter().filter(|a| retain_stmt(cx, &**a)).collect();
        let resulting_stmts = resulting_stmts.into_iter()
            .flat_map(|stmt| cx.fold_stmt(stmt).into_iter())
            .collect();
        let filtered_view_items = view_items.into_iter().filter_map(|a| {
            filter_view_item(cx, a).map(|x| cx.fold_view_item(x))
        }).collect();
        ast::Block {
            id: id,
            view_items: filtered_view_items,
            stmts: resulting_stmts,
            expr: expr.map(|x| cx.fold_expr(x)),
            rules: rules,
            span: span,
        }
    })
}

fn fold_expr(cx: &mut Context, expr: P<ast::Expr>) -> P<ast::Expr> {
    expr.map(|ast::Expr {id, span, node}| {
        fold::noop_fold_expr(ast::Expr {
            id: id,
            node: match node {
                ast::ExprMatch(m, arms, source) => {
                    ast::ExprMatch(m, arms.into_iter()
                                        .filter(|a| (cx.in_cfg)(a.attrs.as_slice()))
                                        .collect(), source)
                }
                _ => node
            },
            span: span
        }, cx)
    })
}

fn item_in_cfg(cx: &mut Context, item: &ast::Item) -> bool {
    return (cx.in_cfg)(item.attrs.as_slice());
}

fn foreign_item_in_cfg(cx: &mut Context, item: &ast::ForeignItem) -> bool {
    return (cx.in_cfg)(item.attrs.as_slice());
}

fn view_item_in_cfg(cx: &mut Context, item: &ast::ViewItem) -> bool {
    return (cx.in_cfg)(item.attrs.as_slice());
}

fn trait_method_in_cfg(cx: &mut Context, meth: &ast::TraitItem) -> bool {
    match *meth {
        ast::RequiredMethod(ref meth) => (cx.in_cfg)(meth.attrs.as_slice()),
        ast::ProvidedMethod(ref meth) => (cx.in_cfg)(meth.attrs.as_slice()),
        ast::TypeTraitItem(ref typ) => (cx.in_cfg)(typ.attrs.as_slice()),
    }
}

fn impl_item_in_cfg(cx: &mut Context, impl_item: &ast::ImplItem) -> bool {
    match *impl_item {
        ast::MethodImplItem(ref meth) => (cx.in_cfg)(meth.attrs.as_slice()),
        ast::TypeImplItem(ref typ) => (cx.in_cfg)(typ.attrs.as_slice()),
    }
}

// Determine if an item should be translated in the current crate
// configuration based on the item's attributes
fn in_cfg(diagnostic: &SpanHandler, cfg: &[P<ast::MetaItem>], attrs: &[ast::Attribute]) -> bool {
    let mut in_cfg = false;
    let mut seen_cfg = false;
    for attr in attrs.iter() {
        let mis = match attr.node.value.node {
            ast::MetaList(_, ref mis) if attr.check_name("cfg") => mis,
            _ => continue
        };

        // NOTE: turn on after snapshot
        /*
        if mis.len() != 1 {
            diagnostic.span_warn(attr.span, "The use of multiple cfgs in the top level of \
                                             `#[cfg(..)]` is deprecated. Change `#[cfg(a, b)]` to \
                                             `#[cfg(all(a, b))]`.");
        }

        if seen_cfg {
            diagnostic.span_warn(attr.span, "The semantics of multiple `#[cfg(..)]` attributes on \
                                             same item are changing from the union of the cfgs to \
                                             the intersection of the cfgs. Change `#[cfg(a)] \
                                             #[cfg(b)]` to `#[cfg(any(a, b))]`.");
        }
        */

        seen_cfg = true;
        in_cfg |= mis.iter().all(|mi| attr::cfg_matches(diagnostic, cfg, &**mi));
    }
    in_cfg | !seen_cfg
}

