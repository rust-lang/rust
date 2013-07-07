// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::option;
use syntax::{ast, fold, attr};

type in_cfg_pred = @fn(attrs: &[ast::attribute]) -> bool;

struct Context {
    in_cfg: in_cfg_pred
}

// Support conditional compilation by transforming the AST, stripping out
// any items that do not belong in the current configuration
pub fn strip_unconfigured_items(crate: @ast::crate) -> @ast::crate {
    do strip_items(crate) |attrs| {
        in_cfg(crate.node.config, attrs)
    }
}

pub fn strip_items(crate: &ast::crate, in_cfg: in_cfg_pred)
    -> @ast::crate {

    let ctxt = @Context { in_cfg: in_cfg };

    let precursor = @fold::AstFoldFns {
          fold_mod: |a,b| fold_mod(ctxt, a, b),
          fold_block: fold::wrap(|a,b| fold_block(ctxt, a, b) ),
          fold_foreign_mod: |a,b| fold_foreign_mod(ctxt, a, b),
          fold_item_underscore: |a,b| {
            // Bad copy.
            fold_item_underscore(ctxt, copy a, b)
          },
          .. *fold::default_ast_fold()};

    let fold = fold::make_fold(precursor);
    @fold.fold_crate(crate)
}

fn filter_item(cx: @Context, item: @ast::item) ->
   Option<@ast::item> {
    if item_in_cfg(cx, item) { option::Some(item) } else { option::None }
}

fn filter_view_item<'r>(cx: @Context, view_item: &'r ast::view_item)-> Option<&'r ast::view_item> {
    if view_item_in_cfg(cx, view_item) {
        option::Some(view_item)
    } else {
        option::None
    }
}

fn fold_mod(cx: @Context, m: &ast::_mod, fld: @fold::ast_fold) -> ast::_mod {
    let filtered_items = do  m.items.iter().filter_map |a| {
        filter_item(cx, *a).chain(|x| fld.fold_item(x))
    }.collect();
    let filtered_view_items = do m.view_items.iter().filter_map |a| {
        filter_view_item(cx, a).map(|&x| fld.fold_view_item(x))
    }.collect();
    ast::_mod {
        view_items: filtered_view_items,
        items: filtered_items
    }
}

fn filter_foreign_item(cx: @Context, item: @ast::foreign_item) ->
   Option<@ast::foreign_item> {
    if foreign_item_in_cfg(cx, item) {
        option::Some(item)
    } else { option::None }
}

fn fold_foreign_mod(
    cx: @Context,
    nm: &ast::foreign_mod,
    fld: @fold::ast_fold
) -> ast::foreign_mod {
    let filtered_items = nm.items.iter().filter_map(|a| filter_foreign_item(cx, *a)).collect();
    let filtered_view_items = do nm.view_items.iter().filter_map |a| {
        filter_view_item(cx, a).map(|&x| fld.fold_view_item(x))
    }.collect();
    ast::foreign_mod {
        sort: nm.sort,
        abis: nm.abis,
        view_items: filtered_view_items,
        items: filtered_items
    }
}

fn fold_item_underscore(cx: @Context, item: &ast::item_,
                        fld: @fold::ast_fold) -> ast::item_ {
    let item = match *item {
        ast::item_impl(ref a, ref b, ref c, ref methods) => {
            let methods = methods.iter().filter(|m| method_in_cfg(cx, **m))
                .transform(|x| *x).collect();
            ast::item_impl(/*bad*/ copy *a, /*bad*/ copy *b, /*bad*/ copy *c, methods)
        }
        ast::item_trait(ref a, ref b, ref methods) => {
            let methods = methods.iter().filter(|m| trait_method_in_cfg(cx, *m) )
                .transform(|x| /* bad */copy *x).collect();
            ast::item_trait(/*bad*/copy *a, /*bad*/copy *b, methods)
        }
        ref item => /*bad*/ copy *item
    };

    fold::noop_fold_item_underscore(&item, fld)
}

fn filter_stmt(cx: @Context, stmt: @ast::stmt) ->
   Option<@ast::stmt> {
    match stmt.node {
      ast::stmt_decl(decl, _) => {
        match decl.node {
          ast::decl_item(item) => {
            if item_in_cfg(cx, item) {
                option::Some(stmt)
            } else { option::None }
          }
          _ => option::Some(stmt)
        }
      }
      _ => option::Some(stmt)
    }
}

fn fold_block(
    cx: @Context,
    b: &ast::blk_,
    fld: @fold::ast_fold
) -> ast::blk_ {
    let resulting_stmts = do b.stmts.iter().filter_map |a| {
        filter_stmt(cx, *a).chain(|stmt| fld.fold_stmt(stmt))
    }.collect();
    let filtered_view_items = do b.view_items.iter().filter_map |a| {
        filter_view_item(cx, a).map(|&x| fld.fold_view_item(x))
    }.collect();
    ast::blk_ {
        view_items: filtered_view_items,
        stmts: resulting_stmts,
        expr: b.expr.map(|x| fld.fold_expr(*x)),
        id: b.id,
        rules: b.rules,
    }
}

fn item_in_cfg(cx: @Context, item: @ast::item) -> bool {
    return (cx.in_cfg)(/*bad*/copy item.attrs);
}

fn foreign_item_in_cfg(cx: @Context, item: @ast::foreign_item) -> bool {
    return (cx.in_cfg)(/*bad*/copy item.attrs);
}

fn view_item_in_cfg(cx: @Context, item: &ast::view_item) -> bool {
    return (cx.in_cfg)(item.attrs);
}

fn method_in_cfg(cx: @Context, meth: @ast::method) -> bool {
    return (cx.in_cfg)(/*bad*/copy meth.attrs);
}

fn trait_method_in_cfg(cx: @Context, meth: &ast::trait_method) -> bool {
    match *meth {
        ast::required(ref meth) => (cx.in_cfg)(/*bad*/copy meth.attrs),
        ast::provided(@ref meth) => (cx.in_cfg)(/*bad*/copy meth.attrs)
    }
}

// Determine if an item should be translated in the current crate
// configuration based on the item's attributes
fn in_cfg(cfg: &[@ast::meta_item], attrs: &[ast::attribute]) -> bool {
    metas_in_cfg(cfg, attr::attr_metas(attrs))
}

pub fn metas_in_cfg(cfg: &[@ast::meta_item],
                    metas: &[@ast::meta_item]) -> bool {
    // The "cfg" attributes on the item
    let cfg_metas = attr::find_meta_items_by_name(metas, "cfg");

    // Pull the inner meta_items from the #[cfg(meta_item, ...)]  attributes,
    // so we can match against them. This is the list of configurations for
    // which the item is valid
    let cfg_metas = cfg_metas.consume_iter()
        .filter_map(|i| attr::get_meta_item_list(i))
        .collect::<~[~[@ast::meta_item]]>();

    if cfg_metas.iter().all(|c| c.is_empty()) { return true; }

    cfg_metas.iter().any_(|cfg_meta| {
        cfg_meta.iter().all(|cfg_mi| {
            match cfg_mi.node {
                ast::meta_list(s, ref it) if "not" == s
                    => it.iter().all(|mi| !attr::contains(cfg, *mi)),
                _ => attr::contains(cfg, *cfg_mi)
            }
        })
    })
}
