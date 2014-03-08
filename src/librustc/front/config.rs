// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::vec_ng::Vec;
use syntax::fold::Folder;
use syntax::{ast, fold, attr};
use syntax::codemap;

struct Context<'a> {
    in_cfg: 'a |attrs: &[ast::Attribute]| -> bool,
}

// Support conditional compilation by transforming the AST, stripping out
// any items that do not belong in the current configuration
pub fn strip_unconfigured_items(krate: ast::Crate) -> ast::Crate {
    let config = krate.config.clone();
    strip_items(krate, |attrs| in_cfg(config.as_slice(), attrs))
}

impl<'a> fold::Folder for Context<'a> {
    fn fold_mod(&mut self, module: &ast::Mod) -> ast::Mod {
        fold_mod(self, module)
    }
    fn fold_block(&mut self, block: ast::P<ast::Block>) -> ast::P<ast::Block> {
        fold_block(self, block)
    }
    fn fold_foreign_mod(&mut self, foreign_mod: &ast::ForeignMod) -> ast::ForeignMod {
        fold_foreign_mod(self, foreign_mod)
    }
    fn fold_item_underscore(&mut self, item: &ast::Item_) -> ast::Item_ {
        fold_item_underscore(self, item)
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

fn filter_view_item<'r>(cx: &Context, view_item: &'r ast::ViewItem)
                        -> Option<&'r ast::ViewItem> {
    if view_item_in_cfg(cx, view_item) {
        Some(view_item)
    } else {
        None
    }
}

fn fold_mod(cx: &mut Context, m: &ast::Mod) -> ast::Mod {
    let filtered_items: Vec<&@ast::Item> = m.items.iter()
            .filter(|&a| item_in_cfg(cx, *a))
            .collect();
    let flattened_items = filtered_items.move_iter()
            .flat_map(|&x| cx.fold_item(x).move_iter())
            .collect();
    let filtered_view_items = m.view_items.iter().filter_map(|a| {
        filter_view_item(cx, a).map(|x| cx.fold_view_item(x))
    }).collect();
    ast::Mod {
        view_items: filtered_view_items,
        items: flattened_items
    }
}

fn filter_foreign_item(cx: &Context, item: @ast::ForeignItem)
                       -> Option<@ast::ForeignItem> {
    if foreign_item_in_cfg(cx, item) {
        Some(item)
    } else {
        None
    }
}

fn fold_foreign_mod(cx: &mut Context, nm: &ast::ForeignMod) -> ast::ForeignMod {
    let filtered_items = nm.items
                           .iter()
                           .filter_map(|a| filter_foreign_item(cx, *a))
                           .collect();
    let filtered_view_items = nm.view_items.iter().filter_map(|a| {
        filter_view_item(cx, a).map(|x| cx.fold_view_item(x))
    }).collect();
    ast::ForeignMod {
        abis: nm.abis,
        view_items: filtered_view_items,
        items: filtered_items
    }
}

fn fold_item_underscore(cx: &mut Context, item: &ast::Item_) -> ast::Item_ {
    let item = match *item {
        ast::ItemImpl(ref a, ref b, c, ref methods) => {
            let methods = methods.iter().filter(|m| method_in_cfg(cx, **m))
                .map(|x| *x).collect();
            ast::ItemImpl((*a).clone(), (*b).clone(), c, methods)
        }
        ast::ItemTrait(ref a, ref b, ref methods) => {
            let methods = methods.iter()
                                 .filter(|m| trait_method_in_cfg(cx, *m) )
                                 .map(|x| (*x).clone())
                                 .collect();
            ast::ItemTrait((*a).clone(), (*b).clone(), methods)
        }
        ast::ItemStruct(def, ref generics) => {
            ast::ItemStruct(fold_struct(cx, def), generics.clone())
        }
        ast::ItemEnum(ref def, ref generics) => {
            let mut variants = def.variants.iter().map(|c| c.clone()).
            filter_map(|v| {
                if !(cx.in_cfg)(v.node.attrs.as_slice()) {
                    None
                } else {
                    Some(match v.node.kind {
                                ast::TupleVariantKind(..) => v,
                                ast::StructVariantKind(def) => {
                                    let def = fold_struct(cx, def);
                                    @codemap::Spanned {
                                        node: ast::Variant_ {
                                            kind: ast::StructVariantKind(def),
                                            ..v.node.clone()
                                        },
                                        ..*v
                                    }
                                }
                            })
                    }
                });
            ast::ItemEnum(ast::EnumDef {
                variants: variants.collect(),
            }, generics.clone())
        }
        ref item => item.clone(),
    };

    fold::noop_fold_item_underscore(&item, cx)
}

fn fold_struct(cx: &Context, def: &ast::StructDef) -> @ast::StructDef {
    let mut fields = def.fields.iter().map(|c| c.clone()).filter(|m| {
        (cx.in_cfg)(m.node.attrs.as_slice())
    });
    @ast::StructDef {
        fields: fields.collect(),
        ctor_id: def.ctor_id,
    }
}

fn retain_stmt(cx: &Context, stmt: @ast::Stmt) -> bool {
    match stmt.node {
      ast::StmtDecl(decl, _) => {
        match decl.node {
          ast::DeclItem(item) => {
            item_in_cfg(cx, item)
          }
          _ => true
        }
      }
      _ => true
    }
}

fn fold_block(cx: &mut Context, b: ast::P<ast::Block>) -> ast::P<ast::Block> {
    let resulting_stmts: Vec<&@ast::Stmt> =
        b.stmts.iter().filter(|&a| retain_stmt(cx, *a)).collect();
    let resulting_stmts = resulting_stmts.move_iter()
        .flat_map(|&stmt| cx.fold_stmt(stmt).move_iter())
        .collect();
    let filtered_view_items = b.view_items.iter().filter_map(|a| {
        filter_view_item(cx, a).map(|x| cx.fold_view_item(x))
    }).collect();
    ast::P(ast::Block {
        view_items: filtered_view_items,
        stmts: resulting_stmts,
        expr: b.expr.map(|x| cx.fold_expr(x)),
        id: b.id,
        rules: b.rules,
        span: b.span,
    })
}

fn item_in_cfg(cx: &Context, item: &ast::Item) -> bool {
    return (cx.in_cfg)(item.attrs.as_slice());
}

fn foreign_item_in_cfg(cx: &Context, item: &ast::ForeignItem) -> bool {
    return (cx.in_cfg)(item.attrs.as_slice());
}

fn view_item_in_cfg(cx: &Context, item: &ast::ViewItem) -> bool {
    return (cx.in_cfg)(item.attrs.as_slice());
}

fn method_in_cfg(cx: &Context, meth: &ast::Method) -> bool {
    return (cx.in_cfg)(meth.attrs.as_slice());
}

fn trait_method_in_cfg(cx: &Context, meth: &ast::TraitMethod) -> bool {
    match *meth {
        ast::Required(ref meth) => (cx.in_cfg)(meth.attrs.as_slice()),
        ast::Provided(meth) => (cx.in_cfg)(meth.attrs.as_slice())
    }
}

// Determine if an item should be translated in the current crate
// configuration based on the item's attributes
fn in_cfg(cfg: &[@ast::MetaItem], attrs: &[ast::Attribute]) -> bool {
    attr::test_cfg(cfg, attrs.iter().map(|x| *x))
}

