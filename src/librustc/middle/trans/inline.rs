// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use metadata::csearch;
use middle::astencode;
use middle::trans::base::{push_ctxt, impl_self, no_self};
use middle::trans::base::{trans_item, get_item_val, trans_fn};
use middle::trans::common::*;
use middle::ty;
use util::ppaux::ty_to_str;

use std::vec;
use syntax::ast;
use syntax::ast_map::path_name;
use syntax::ast_util::local_def;

pub fn maybe_instantiate_inline(ccx: @mut CrateContext, fn_id: ast::DefId)
    -> ast::DefId {
    let _icx = push_ctxt("maybe_instantiate_inline");
    match ccx.external.find(&fn_id) {
        Some(&Some(node_id)) => {
            // Already inline
            debug!("maybe_instantiate_inline(%s): already inline as node id %d",
                   ty::item_path_str(ccx.tcx, fn_id), node_id);
            return local_def(node_id);
        }
        Some(&None) => {
            return fn_id; // Not inlinable
        }
        None => {
            // Not seen yet
        }
    }

    let csearch_result =
        csearch::maybe_get_item_ast(
            ccx.tcx, fn_id,
            |a,b,c,d| {
                astencode::decode_inlined_item(a, b, ccx.maps, c.clone(), d)
            });
    return match csearch_result {
        csearch::not_found => {
            ccx.external.insert(fn_id, None);
            fn_id
        }
        csearch::found(ast::ii_item(item)) => {
            ccx.external.insert(fn_id, Some(item.id));
            ccx.stats.n_inlines += 1;
            trans_item(ccx, item);
            local_def(item.id)
        }
        csearch::found(ast::ii_foreign(item)) => {
          ccx.external.insert(fn_id, Some(item.id));
          local_def(item.id)
        }
        csearch::found_parent(parent_id, ast::ii_item(item)) => {
          ccx.external.insert(parent_id, Some(item.id));
          let mut my_id = 0;
          match item.node {
            ast::item_enum(_, _) => {
              let vs_here = ty::enum_variants(ccx.tcx, local_def(item.id));
              let vs_there = ty::enum_variants(ccx.tcx, parent_id);
              for (here, there) in vs_here.iter().zip(vs_there.iter()) {
                  if there.id == fn_id { my_id = here.id.node; }
                  ccx.external.insert(there.id, Some(here.id.node));
              }
            }
            _ => ccx.sess.bug("maybe_instantiate_inline: item has a \
                               non-enum parent")
          }
          trans_item(ccx, item);
          local_def(my_id)
        }
        csearch::found_parent(_, _) => {
            ccx.sess.bug("maybe_get_item_ast returned a found_parent \
             with a non-item parent");
        }
        csearch::found(ast::ii_method(impl_did, is_provided, mth)) => {
          ccx.stats.n_inlines += 1;
          ccx.external.insert(fn_id, Some(mth.id));
          // If this is a default method, we can't look up the
          // impl type. But we aren't going to translate anyways, so don't.
          if is_provided { return local_def(mth.id); }

            let impl_tpt = ty::lookup_item_type(ccx.tcx, impl_did);
            let num_type_params =
                impl_tpt.generics.type_param_defs.len() +
                mth.generics.ty_params.len();

          if num_type_params == 0 {
              let llfn = get_item_val(ccx, mth.id);
              let path = vec::append(
                  ty::item_path(ccx.tcx, impl_did),
                  [path_name(mth.ident)]);
              let self_kind = match mth.explicit_self.node {
                  ast::sty_static => no_self,
                  _ => {
                      let self_ty = ty::node_id_to_type(ccx.tcx,
                                                        mth.self_id);
                      debug!("calling inline trans_fn with self_ty %s",
                             ty_to_str(ccx.tcx, self_ty));
                      match mth.explicit_self.node {
                          ast::sty_value => impl_self(self_ty, ty::ByRef),
                          _ => impl_self(self_ty, ty::ByCopy),
                      }
                  }
              };
              trans_fn(ccx,
                       path,
                       &mth.decl,
                       &mth.body,
                       llfn,
                       self_kind,
                       None,
                       mth.id,
                       []);
          }
          local_def(mth.id)
        }
    };
}
