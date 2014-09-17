// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::{AvailableExternallyLinkage, InternalLinkage, SetLinkage};
use metadata::csearch;
use middle::astencode;
use middle::trans::base::{push_ctxt, trans_item, get_item_val, trans_fn};
use middle::trans::common::*;
use middle::ty;

use syntax::ast;
use syntax::ast_util::{local_def, PostExpansionMethod};
use syntax::ast_util;

fn instantiate_inline(ccx: &CrateContext, fn_id: ast::DefId)
    -> Option<ast::DefId> {
    let _icx = push_ctxt("maybe_instantiate_inline");
    match ccx.external().borrow().find(&fn_id) {
        Some(&Some(node_id)) => {
            // Already inline
            debug!("maybe_instantiate_inline({}): already inline as node id {}",
                   ty::item_path_str(ccx.tcx(), fn_id), node_id);
            return Some(local_def(node_id));
        }
        Some(&None) => {
            return None; // Not inlinable
        }
        None => {
            // Not seen yet
        }
    }

    let csearch_result =
        csearch::maybe_get_item_ast(
            ccx.tcx(), fn_id,
            |a,b,c,d| astencode::decode_inlined_item(a, b, c, d));

    let inline_def = match csearch_result {
        csearch::not_found => {
            ccx.external().borrow_mut().insert(fn_id, None);
            return None;
        }
        csearch::found(&ast::IIItem(ref item)) => {
            ccx.external().borrow_mut().insert(fn_id, Some(item.id));
            ccx.external_srcs().borrow_mut().insert(item.id, fn_id);

            ccx.stats().n_inlines.set(ccx.stats().n_inlines.get() + 1);
            trans_item(ccx, &**item);

            let linkage = match item.node {
                ast::ItemFn(_, _, _, ref generics, _) => {
                    if generics.is_type_parameterized() {
                        // Generics have no symbol, so they can't be given any
                        // linkage.
                        None
                    } else {
                        if ccx.sess().opts.cg.codegen_units == 1 {
                            // We could use AvailableExternallyLinkage here,
                            // but InternalLinkage allows LLVM to optimize more
                            // aggressively (at the cost of sometimes
                            // duplicating code).
                            Some(InternalLinkage)
                        } else {
                            // With multiple compilation units, duplicated code
                            // is more of a problem.  Also, `codegen_units > 1`
                            // means the user is okay with losing some
                            // performance.
                            Some(AvailableExternallyLinkage)
                        }
                    }
                }
                ast::ItemStatic(_, mutbl, _) => {
                    if !ast_util::static_has_significant_address(mutbl, item.attrs.as_slice()) {
                        // Inlined static items use internal linkage when
                        // possible, so that LLVM will coalesce globals with
                        // identical initializers.  (It only does this for
                        // globals with unnamed_addr and either internal or
                        // private linkage.)
                        Some(InternalLinkage)
                    } else {
                        // The address is significant, so we can't create an
                        // internal copy of the static.  (The copy would have a
                        // different address from the original.)
                        Some(AvailableExternallyLinkage)
                    }
                }
                _ => unreachable!(),
            };

            match linkage {
                Some(linkage) => {
                    let g = get_item_val(ccx, item.id);
                    SetLinkage(g, linkage);
                }
                None => {}
            }

            local_def(item.id)
        }
        csearch::found(&ast::IIForeign(ref item)) => {
            ccx.external().borrow_mut().insert(fn_id, Some(item.id));
            ccx.external_srcs().borrow_mut().insert(item.id, fn_id);
            local_def(item.id)
        }
        csearch::found_parent(parent_id, &ast::IIItem(ref item)) => {
            ccx.external().borrow_mut().insert(parent_id, Some(item.id));
            ccx.external_srcs().borrow_mut().insert(item.id, parent_id);

          let mut my_id = 0;
          match item.node {
            ast::ItemEnum(_, _) => {
              let vs_here = ty::enum_variants(ccx.tcx(), local_def(item.id));
              let vs_there = ty::enum_variants(ccx.tcx(), parent_id);
              for (here, there) in vs_here.iter().zip(vs_there.iter()) {
                  if there.id == fn_id { my_id = here.id.node; }
                  ccx.external().borrow_mut().insert(there.id, Some(here.id.node));
              }
            }
            ast::ItemStruct(ref struct_def, _) => {
              match struct_def.ctor_id {
                None => {}
                Some(ctor_id) => {
                    ccx.external().borrow_mut().insert(fn_id, Some(ctor_id));
                    my_id = ctor_id;
                }
              }
            }
            _ => ccx.sess().bug("maybe_instantiate_inline: item has a \
                                 non-enum, non-struct parent")
          }
          trans_item(ccx, &**item);
          local_def(my_id)
        }
        csearch::found_parent(_, _) => {
            ccx.sess().bug("maybe_get_item_ast returned a found_parent \
             with a non-item parent");
        }
        csearch::found(&ast::IITraitItem(_, ref trait_item)) => {
            match *trait_item {
                ast::RequiredMethod(_) => ccx.sess().bug("found RequiredMethod IITraitItem"),
                ast::ProvidedMethod(ref mth) => {
                    ccx.external().borrow_mut().insert(fn_id, Some(mth.id));
                    ccx.external_srcs().borrow_mut().insert(mth.id, fn_id);

                    ccx.stats().n_inlines.set(ccx.stats().n_inlines.get() + 1);

                    // If this is a default method, we can't look up the
                    // impl type. But we aren't going to translate anyways, so
                    // don't.
                    local_def(mth.id)
                }
                ast::TypeTraitItem(_) => {
                    ccx.sess().bug("found TypeTraitItem IITraitItem")
                }
            }
        }
        csearch::found(&ast::IIImplItem(impl_did, ref impl_item)) => {
            match *impl_item {
                ast::MethodImplItem(ref mth) => {
                    ccx.external().borrow_mut().insert(fn_id, Some(mth.id));
                    ccx.external_srcs().borrow_mut().insert(mth.id, fn_id);

                    ccx.stats().n_inlines.set(ccx.stats().n_inlines.get() + 1);

                    let impl_tpt = ty::lookup_item_type(ccx.tcx(), impl_did);
                    let unparameterized = impl_tpt.generics.types.is_empty() &&
                            mth.pe_generics().ty_params.is_empty();

                    if unparameterized {
                        let llfn = get_item_val(ccx, mth.id);
                        trans_fn(ccx,
                                 &*mth.pe_fn_decl(),
                                 &*mth.pe_body(),
                                 llfn,
                                 &param_substs::empty(),
                                 mth.id,
                                 []);
                        // Use InternalLinkage so LLVM can optimize more
                        // aggressively.
                        SetLinkage(llfn, InternalLinkage);
                    }
                    local_def(mth.id)
                }
                ast::TypeImplItem(_) => {
                    ccx.sess().bug("found TypeImplItem IIImplItem")
                }
            }
        }
    };

    return Some(inline_def);
}

pub fn get_local_instance(ccx: &CrateContext, fn_id: ast::DefId)
    -> Option<ast::DefId> {
    if fn_id.krate == ast::LOCAL_CRATE {
        Some(fn_id)
    } else {
        instantiate_inline(ccx, fn_id)
    }
}

pub fn maybe_instantiate_inline(ccx: &CrateContext, fn_id: ast::DefId) -> ast::DefId {
    get_local_instance(ccx, fn_id).unwrap_or(fn_id)
}
