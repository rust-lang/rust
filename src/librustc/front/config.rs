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

struct Context<'self> {
    in_cfg: &'self fn(attrs: &[ast::Attribute]) -> bool,
}

// Support conditional compilation by transforming the AST, stripping out
// any items that do not belong in the current configuration
pub fn strip_unconfigured_items(crate: ast::Crate) -> ast::Crate {
    let config = crate.config.clone();
    do strip_items(crate) |attrs| {
        in_cfg(config, attrs)
    }
}

impl<'self> fold::ast_fold for Context<'self> {
    fn fold_mod(&self, module: &ast::_mod) -> ast::_mod {
        fold_mod(self, module)
    }
    fn fold_block(&self, block: &ast::Block) -> ast::Block {
        fold_block(self, block)
    }
    fn fold_foreign_mod(&self, foreign_module: &ast::foreign_mod)
                        -> ast::foreign_mod {
        fold_foreign_mod(self, foreign_module)
    }
    fn fold_item_underscore(&self, item: &ast::item_) -> ast::item_ {
        fold_item_underscore(self, item)
    }
}

pub fn strip_items(crate: ast::Crate,
                   in_cfg: &fn(attrs: &[ast::Attribute]) -> bool)
                   -> ast::Crate {
    let ctxt = Context {
        in_cfg: in_cfg,
    };
    ctxt.fold_crate(crate)
}

fn filter_item(cx: &Context, item: @ast::item) -> Option<@ast::item> {
    if item_in_cfg(cx, item) {
        Some(item)
    } else {
        None
    }
}

fn filter_view_item<'r>(cx: &Context, view_item: &'r ast::view_item)
                        -> Option<&'r ast::view_item> {
    if view_item_in_cfg(cx, view_item) {
        Some(view_item)
    } else {
        None
    }
}

fn fold_mod(cx: &Context, m: &ast::_mod) -> ast::_mod {
    let filtered_items = do m.items.iter().filter_map |a| {
        filter_item(cx, *a).and_then(|x| cx.fold_item(x))
    }.collect();
    let filtered_view_items = do m.view_items.iter().filter_map |a| {
        do filter_view_item(cx, a).map |x| {
            cx.fold_view_item(x)
        }
    }.collect();
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

fn fold_foreign_mod(cx: &Context, nm: &ast::foreign_mod) -> ast::foreign_mod {
    let filtered_items = nm.items
                           .iter()
                           .filter_map(|a| filter_foreign_item(cx, *a))
                           .collect();
    let filtered_view_items = do nm.view_items.iter().filter_map |a| {
        do filter_view_item(cx, a).map |x| {
            cx.fold_view_item(x)
        }
    }.collect();
    ast::foreign_mod {
        abis: nm.abis,
        view_items: filtered_view_items,
        items: filtered_items
    }
}

fn fold_item_underscore(cx: &Context, item: &ast::item_) -> ast::item_ {
    let item = match *item {
        ast::item_impl(ref a, ref b, ref c, ref methods) => {
            let methods = methods.iter().filter(|m| method_in_cfg(cx, **m))
                .map(|x| *x).collect();
            ast::item_impl((*a).clone(), (*b).clone(), (*c).clone(), methods)
        }
        ast::item_trait(ref a, ref b, ref methods) => {
            let methods = methods.iter()
                                 .filter(|m| trait_method_in_cfg(cx, *m) )
                                 .map(|x| (*x).clone())
                                 .collect();
            ast::item_trait((*a).clone(), (*b).clone(), methods)
        }
        ref item => (*item).clone(),
    };

    fold::noop_fold_item_underscore(&item, cx)
}

fn filter_stmt(cx: &Context, stmt: @ast::Stmt) -> Option<@ast::Stmt> {
    match stmt.node {
      ast::StmtDecl(decl, _) => {
        match decl.node {
          ast::DeclItem(item) => {
            if item_in_cfg(cx, item) {
                Some(stmt)
            } else {
                None
            }
          }
          _ => Some(stmt)
        }
      }
      _ => Some(stmt),
    }
}

fn fold_block(cx: &Context, b: &ast::Block) -> ast::Block {
    let resulting_stmts = do b.stmts.iter().filter_map |a| {
        filter_stmt(cx, *a).and_then(|stmt| cx.fold_stmt(stmt))
    }.collect();
    let filtered_view_items = do b.view_items.iter().filter_map |a| {
        filter_view_item(cx, a).map(|x| cx.fold_view_item(x))
    }.collect();
    ast::Block {
        view_items: filtered_view_items,
        stmts: resulting_stmts,
        expr: b.expr.map(|x| cx.fold_expr(x)),
        id: b.id,
        rules: b.rules,
        span: b.span,
    }
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

