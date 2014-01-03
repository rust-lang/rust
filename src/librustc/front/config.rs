// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use syntax::fold::ast_fold;
use syntax::{ast, fold, attr};
use syntax::codemap;

struct Context<'a> {
    in_cfg: 'a |attrs: &[ast::Attribute]| -> bool,
}

// Support conditional compilation by transforming the AST, stripping out
// any items that do not belong in the current configuration
pub fn strip_unconfigured_items(crate: ast::Crate) -> ast::Crate {
    let config = crate.config.clone();
    strip_items(crate, |attrs| in_cfg(config, attrs))
}

impl<'a> fold::ast_fold for Context<'a> {
    fn fold_mod(&mut self, module: &ast::_mod) -> ast::_mod {
        fold_mod(self, module)
    }
    fn fold_block(&mut self, block: ast::P<ast::Block>) -> ast::P<ast::Block> {
        fold_block(self, block)
    }
    fn fold_foreign_mod(&mut self, foreign_module: &ast::foreign_mod)
                        -> ast::foreign_mod {
        fold_foreign_mod(self, foreign_module)
    }
    fn fold_item_underscore(&mut self, item: &ast::item_) -> ast::item_ {
        fold_item_underscore(self, item)
    }
}

pub fn strip_items(crate: ast::Crate,
                   in_cfg: |attrs: &[ast::Attribute]| -> bool)
                   -> ast::Crate {
    let mut ctxt = Context {
        in_cfg: in_cfg,
    };
    ctxt.fold_crate(crate)
}

fn filter_view_item<'r>(cx: &Context, view_item: &'r ast::view_item)
                        -> Option<&'r ast::view_item> {
    if view_item_in_cfg(cx, view_item) {
        Some(view_item)
    } else {
        None
    }
}

fn fold_mod(cx: &mut Context, m: &ast::_mod) -> ast::_mod {
    let filtered_items = m.items.iter()
            .filter(|&a| item_in_cfg(cx, *a))
            .flat_map(|&x| cx.fold_item(x).move_iter())
            .collect();
    let filtered_view_items = m.view_items.iter().filter_map(|a| {
        filter_view_item(cx, a).map(|x| cx.fold_view_item(x))
    }).collect();
    ast::_mod {
        view_items: filtered_view_items,
        items: filtered_items
    }
}

fn filter_foreign_item(cx: &Context, item: @ast::foreign_item)
                       -> Option<@ast::foreign_item> {
    if foreign_item_in_cfg(cx, item) {
        Some(item)
    } else {
        None
    }
}

fn fold_foreign_mod(cx: &mut Context, nm: &ast::foreign_mod) -> ast::foreign_mod {
    let filtered_items = nm.items
                           .iter()
                           .filter_map(|a| filter_foreign_item(cx, *a))
                           .collect();
    let filtered_view_items = nm.view_items.iter().filter_map(|a| {
        filter_view_item(cx, a).map(|x| cx.fold_view_item(x))
    }).collect();
    ast::foreign_mod {
        abis: nm.abis,
        view_items: filtered_view_items,
        items: filtered_items
    }
}

fn fold_item_underscore(cx: &mut Context, item: &ast::item_) -> ast::item_ {
    let item = match *item {
        ast::item_impl(ref a, ref b, c, ref methods) => {
            let methods = methods.iter().filter(|m| method_in_cfg(cx, **m))
                .map(|x| *x).collect();
            ast::item_impl((*a).clone(), (*b).clone(), c, methods)
        }
        ast::item_trait(ref a, ref b, ref methods) => {
            let methods = methods.iter()
                                 .filter(|m| trait_method_in_cfg(cx, *m) )
                                 .map(|x| (*x).clone())
                                 .collect();
            ast::item_trait((*a).clone(), (*b).clone(), methods)
        }
        ast::item_struct(def, ref generics) => {
            ast::item_struct(fold_struct(cx, def), generics.clone())
        }
        ast::item_enum(ref def, ref generics) => {
            let mut variants = def.variants.iter().map(|c| c.clone()).filter(|m| {
                (cx.in_cfg)(m.node.attrs)
            }).map(|v| {
                match v.node.kind {
                    ast::tuple_variant_kind(..) => v,
                    ast::struct_variant_kind(def) => {
                        let def = fold_struct(cx, def);
                        @codemap::Spanned {
                            node: ast::variant_ {
                                kind: ast::struct_variant_kind(def),
                                ..v.node.clone()
                            },
                            ..*v
                        }
                    }
                }
            });
            ast::item_enum(ast::enum_def {
                variants: variants.collect(),
            }, generics.clone())
        }
        ref item => item.clone(),
    };

    fold::noop_fold_item_underscore(&item, cx)
}

fn fold_struct(cx: &Context, def: &ast::struct_def) -> @ast::struct_def {
    let mut fields = def.fields.iter().map(|c| c.clone()).filter(|m| {
        (cx.in_cfg)(m.node.attrs)
    });
    @ast::struct_def {
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
    let resulting_stmts = b.stmts.iter()
            .filter(|&a| retain_stmt(cx, *a))
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

fn item_in_cfg(cx: &Context, item: @ast::item) -> bool {
    return (cx.in_cfg)(item.attrs);
}

fn foreign_item_in_cfg(cx: &Context, item: @ast::foreign_item) -> bool {
    return (cx.in_cfg)(item.attrs);
}

fn view_item_in_cfg(cx: &Context, item: &ast::view_item) -> bool {
    return (cx.in_cfg)(item.attrs);
}

fn method_in_cfg(cx: &Context, meth: @ast::method) -> bool {
    return (cx.in_cfg)(meth.attrs);
}

fn trait_method_in_cfg(cx: &Context, meth: &ast::trait_method) -> bool {
    match *meth {
        ast::required(ref meth) => (cx.in_cfg)(meth.attrs),
        ast::provided(@ref meth) => (cx.in_cfg)(meth.attrs)
    }
}

// Determine if an item should be translated in the current crate
// configuration based on the item's attributes
fn in_cfg(cfg: &[@ast::MetaItem], attrs: &[ast::Attribute]) -> bool {
    attr::test_cfg(cfg, attrs.iter().map(|x| *x))
}

