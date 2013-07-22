// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi::AbiSet;
use ast::*;
use ast;
use ast_util::{inlined_item_utils, stmt_id};
use ast_util;
use codemap;
use diagnostic::span_handler;
use parse::token::ident_interner;
use print::pprust;
use visit;
use syntax::parse::token::special_idents;

use std::hashmap::HashMap;
use std::vec;

#[deriving(Clone, Eq)]
pub enum path_elt {
    path_mod(ident),
    path_name(ident)
}

pub type path = ~[path_elt];

pub fn path_to_str_with_sep(p: &[path_elt], sep: &str, itr: @ident_interner)
                         -> ~str {
    let strs = do p.map |e| {
        match *e {
          path_mod(s) => itr.get(s.name),
          path_name(s) => itr.get(s.name)
        }
    };
    strs.connect(sep)
}

pub fn path_ident_to_str(p: &path, i: ident, itr: @ident_interner) -> ~str {
    if p.is_empty() {
        itr.get(i.name).to_owned()
    } else {
        fmt!("%s::%s", path_to_str(*p, itr), itr.get(i.name))
    }
}

pub fn path_to_str(p: &[path_elt], itr: @ident_interner) -> ~str {
    path_to_str_with_sep(p, "::", itr)
}

pub fn path_elt_to_str(pe: path_elt, itr: @ident_interner) -> ~str {
    match pe {
        path_mod(s) => itr.get(s.name).to_owned(),
        path_name(s) => itr.get(s.name).to_owned()
    }
}

#[deriving(Clone)]
pub enum ast_node {
    node_item(@item, @path),
    node_foreign_item(@foreign_item, AbiSet, visibility, @path),
    node_trait_method(@trait_method, def_id /* trait did */,
                      @path /* path to the trait */),
    node_method(@method, def_id /* impl did */, @path /* path to the impl */),
    node_variant(variant, @item, @path),
    node_expr(@expr),
    node_stmt(@stmt),
    node_arg,
    node_local(ident),
    node_block(Block),
    node_struct_ctor(@struct_def, @item, @path),
    node_callee_scope(@expr)
}

pub type map = @mut HashMap<node_id, ast_node>;

pub struct Ctx {
    map: map,
    path: path,
    diag: @span_handler,
}

pub type vt = visit::vt<@mut Ctx>;

pub fn extend(cx: @mut Ctx, elt: ident) -> @path {
    @(vec::append(cx.path.clone(), [path_name(elt)]))
}

pub fn mk_ast_map_visitor() -> vt {
    return visit::mk_vt(@visit::Visitor {
        visit_item: map_item,
        visit_expr: map_expr,
        visit_stmt: map_stmt,
        visit_fn: map_fn,
        visit_block: map_block,
        visit_pat: map_pat,
        .. *visit::default_visitor()
    });
}

pub fn map_crate(diag: @span_handler, c: &Crate) -> map {
    let cx = @mut Ctx {
        map: @mut HashMap::new(),
        path: ~[],
        diag: diag,
    };
    visit::visit_crate(c, (cx, mk_ast_map_visitor()));
    cx.map
}

// Used for items loaded from external crate that are being inlined into this
// crate.  The `path` should be the path to the item but should not include
// the item itself.
pub fn map_decoded_item(diag: @span_handler,
                        map: map,
                        path: path,
                        ii: &inlined_item) {
    // I believe it is ok for the local IDs of inlined items from other crates
    // to overlap with the local ids from this crate, so just generate the ids
    // starting from 0.  (In particular, I think these ids are only used in
    // alias analysis, which we will not be running on the inlined items, and
    // even if we did I think it only needs an ordering between local
    // variables that are simultaneously in scope).
    let cx = @mut Ctx {
        map: map,
        path: path.clone(),
        diag: diag,
    };
    let v = mk_ast_map_visitor();

    // methods get added to the AST map when their impl is visited.  Since we
    // don't decode and instantiate the impl, but just the method, we have to
    // add it to the table now:
    match *ii {
      ii_item(*) => { /* fallthrough */ }
      ii_foreign(i) => {
        cx.map.insert(i.id, node_foreign_item(i,
                                              AbiSet::Intrinsic(),
                                              i.vis,    // Wrong but OK
                                              @path));
      }
      ii_method(impl_did, is_provided, m) => {
        map_method(impl_did, @path, m, is_provided, cx);
      }
    }

    // visit the item / method contents and add those to the map:
    ii.accept(cx, v);
}

pub fn map_fn(
    fk: &visit::fn_kind,
    decl: &fn_decl,
    body: &Block,
    sp: codemap::span,
    id: node_id,
    (cx,v): (@mut Ctx,
             visit::vt<@mut Ctx>)
) {
    for decl.inputs.iter().advance |a| {
        cx.map.insert(a.id, node_arg);
    }
    visit::visit_fn(fk, decl, body, sp, id, (cx, v));
}

pub fn map_block(b: &Block, (cx,v): (@mut Ctx, visit::vt<@mut Ctx>)) {
    cx.map.insert(b.id, node_block(/* FIXME (#2543) */ (*b).clone()));
    visit::visit_block(b, (cx, v));
}

pub fn map_pat(pat: @pat, (cx,v): (@mut Ctx, visit::vt<@mut Ctx>)) {
    match pat.node {
        pat_ident(_, ref path, _) => {
            // Note: this is at least *potentially* a pattern...
            cx.map.insert(pat.id, node_local(ast_util::path_to_ident(path)));
        }
        _ => ()
    }

    visit::visit_pat(pat, (cx, v));
}

pub fn map_method(impl_did: def_id, impl_path: @path,
                  m: @method, is_provided: bool, cx: @mut Ctx) {
    let entry = if is_provided {
        node_trait_method(@provided(m), impl_did, impl_path)
    } else { node_method(m, impl_did, impl_path) };
    cx.map.insert(m.id, entry);
    cx.map.insert(m.self_id, node_local(special_idents::self_));
}

pub fn map_item(i: @item, (cx, v): (@mut Ctx, visit::vt<@mut Ctx>)) {
    let item_path = @/* FIXME (#2543) */ cx.path.clone();
    cx.map.insert(i.id, node_item(i, item_path));
    match i.node {
        item_impl(_, _, _, ref ms) => {
            let impl_did = ast_util::local_def(i.id);
            for ms.iter().advance |m| {
                map_method(impl_did, extend(cx, i.ident), *m, false, cx);
            }
        }
        item_enum(ref enum_definition, _) => {
            for (*enum_definition).variants.iter().advance |v| {
                cx.map.insert(v.node.id, node_variant(
                    /* FIXME (#2543) */ (*v).clone(),
                    i,
                    extend(cx, i.ident)));
            }
        }
        item_foreign_mod(ref nm) => {
            for nm.items.iter().advance |nitem| {
                // Compute the visibility for this native item.
                let visibility = match nitem.vis {
                    public => public,
                    private => private,
                    inherited => i.vis
                };

                cx.map.insert(nitem.id,
                    node_foreign_item(
                        *nitem,
                        nm.abis,
                        visibility,
                        // FIXME (#2543)
                        if nm.sort == ast::named {
                            extend(cx, i.ident)
                        } else {
                            // Anonymous extern mods go in the parent scope
                            @cx.path.clone()
                        }
                    )
                );
            }
        }
        item_struct(struct_def, _) => {
            map_struct_def(
                struct_def,
                node_item(i, item_path),
                i.ident,
                (cx,
                 v)
            );
        }
        item_trait(_, ref traits, ref methods) => {
            for traits.iter().advance |p| {
                cx.map.insert(p.ref_id, node_item(i, item_path));
            }
            for methods.iter().advance |tm| {
                let id = ast_util::trait_method_to_ty_method(tm).id;
                let d_id = ast_util::local_def(i.id);
                cx.map.insert(
                    id,
                    node_trait_method(@(*tm).clone(), d_id, item_path)
                );
            }
        }
        _ => ()
    }

    match i.node {
        item_mod(_) | item_foreign_mod(_) => {
            cx.path.push(path_mod(i.ident));
        }
        _ => cx.path.push(path_name(i.ident))
    }
    visit::visit_item(i, (cx, v));
    cx.path.pop();
}

pub fn map_struct_def(
    struct_def: @ast::struct_def,
    parent_node: ast_node,
    ident: ast::ident,
    (cx, _v): (@mut Ctx,
               visit::vt<@mut Ctx>)
) {
    let p = extend(cx, ident);
    // If this is a tuple-like struct, register the constructor.
    match struct_def.ctor_id {
        None => {}
        Some(ctor_id) => {
            match parent_node {
                node_item(item, _) => {
                    cx.map.insert(ctor_id,
                                  node_struct_ctor(struct_def, item, p));
                }
                _ => fail!("struct def parent wasn't an item")
            }
        }
    }
}

pub fn map_expr(ex: @expr, (cx,v): (@mut Ctx, visit::vt<@mut Ctx>)) {
    cx.map.insert(ex.id, node_expr(ex));
    // Expressions which are or might be calls:
    {
        let r = ex.get_callee_id();
        for r.iter().advance |callee_id| {
            cx.map.insert(*callee_id, node_callee_scope(ex));
        }
    }
    visit::visit_expr(ex, (cx, v));
}

pub fn map_stmt(stmt: @stmt, (cx,v): (@mut Ctx, visit::vt<@mut Ctx>)) {
    cx.map.insert(stmt_id(stmt), node_stmt(stmt));
    visit::visit_stmt(stmt, (cx, v));
}

pub fn node_id_to_str(map: map, id: node_id, itr: @ident_interner) -> ~str {
    match map.find(&id) {
      None => {
        fmt!("unknown node (id=%d)", id)
      }
      Some(&node_item(item, path)) => {
        let path_str = path_ident_to_str(path, item.ident, itr);
        let item_str = match item.node {
          item_static(*) => ~"static",
          item_fn(*) => ~"fn",
          item_mod(*) => ~"mod",
          item_foreign_mod(*) => ~"foreign mod",
          item_ty(*) => ~"ty",
          item_enum(*) => ~"enum",
          item_struct(*) => ~"struct",
          item_trait(*) => ~"trait",
          item_impl(*) => ~"impl",
          item_mac(*) => ~"macro"
        };
        fmt!("%s %s (id=%?)", item_str, path_str, id)
      }
      Some(&node_foreign_item(item, abi, _, path)) => {
        fmt!("foreign item %s with abi %? (id=%?)",
             path_ident_to_str(path, item.ident, itr), abi, id)
      }
      Some(&node_method(m, _, path)) => {
        fmt!("method %s in %s (id=%?)",
             itr.get(m.ident.name), path_to_str(*path, itr), id)
      }
      Some(&node_trait_method(ref tm, _, path)) => {
        let m = ast_util::trait_method_to_ty_method(&**tm);
        fmt!("method %s in %s (id=%?)",
             itr.get(m.ident.name), path_to_str(*path, itr), id)
      }
      Some(&node_variant(ref variant, _, path)) => {
        fmt!("variant %s in %s (id=%?)",
             itr.get(variant.node.name.name), path_to_str(*path, itr), id)
      }
      Some(&node_expr(expr)) => {
        fmt!("expr %s (id=%?)", pprust::expr_to_str(expr, itr), id)
      }
      Some(&node_callee_scope(expr)) => {
        fmt!("callee_scope %s (id=%?)", pprust::expr_to_str(expr, itr), id)
      }
      Some(&node_stmt(stmt)) => {
        fmt!("stmt %s (id=%?)",
             pprust::stmt_to_str(stmt, itr), id)
      }
      Some(&node_arg) => {
        fmt!("arg (id=%?)", id)
      }
      Some(&node_local(ident)) => {
        fmt!("local (id=%?, name=%s)", id, itr.get(ident.name))
      }
      Some(&node_block(_)) => {
        fmt!("block")
      }
      Some(&node_struct_ctor(*)) => {
        fmt!("struct_ctor")
      }
    }
}

pub fn node_item_query<Result>(items: map, id: node_id,
                               query: &fn(@item) -> Result,
                               error_msg: ~str) -> Result {
    match items.find(&id) {
        Some(&node_item(it, _)) => query(it),
        _ => fail!(error_msg)
    }
}
