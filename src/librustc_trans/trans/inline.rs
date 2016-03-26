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
use middle::cstore::{CrateStore, FoundAst, InlinedItem};
use middle::def_id::DefId;
use middle::subst::Substs;
use trans::base::{push_ctxt, trans_item, trans_fn};
use trans::callee::Callee;
use trans::common::*;

use rustc::dep_graph::DepNode;
use rustc_front::hir;

fn instantiate_inline(ccx: &CrateContext, fn_id: DefId) -> Option<DefId> {
    debug!("instantiate_inline({:?})", fn_id);
    let _icx = push_ctxt("instantiate_inline");
    let tcx = ccx.tcx();
    let _task = tcx.dep_graph.in_task(DepNode::TransInlinedItem(fn_id));

    match ccx.external().borrow().get(&fn_id) {
        Some(&Some(node_id)) => {
            // Already inline
            debug!("instantiate_inline({}): already inline as node id {}",
                   tcx.item_path_str(fn_id), node_id);
            let node_def_id = tcx.map.local_def_id(node_id);
            return Some(node_def_id);
        }
        Some(&None) => {
            return None; // Not inlinable
        }
        None => {
            // Not seen yet
        }
    }

    let inlined = tcx.sess.cstore.maybe_get_item_ast(tcx, fn_id);
    let inline_id = match inlined {
        FoundAst::NotFound => {
            ccx.external().borrow_mut().insert(fn_id, None);
            return None;
        }
        FoundAst::Found(&InlinedItem::Item(ref item)) => {
            ccx.external().borrow_mut().insert(fn_id, Some(item.id));
            ccx.external_srcs().borrow_mut().insert(item.id, fn_id);

            ccx.stats().n_inlines.set(ccx.stats().n_inlines.get() + 1);
            trans_item(ccx, item);

            if let hir::ItemFn(_, _, _, _, ref generics, _) = item.node {
                // Generics have no symbol, so they can't be given any linkage.
                if !generics.is_type_parameterized() {
                    let linkage = if ccx.sess().opts.cg.codegen_units == 1 {
                        // We could use AvailableExternallyLinkage here,
                        // but InternalLinkage allows LLVM to optimize more
                        // aggressively (at the cost of sometimes
                        // duplicating code).
                        InternalLinkage
                    } else {
                        // With multiple compilation units, duplicated code
                        // is more of a problem.  Also, `codegen_units > 1`
                        // means the user is okay with losing some
                        // performance.
                        AvailableExternallyLinkage
                    };
                    let empty_substs = tcx.mk_substs(Substs::empty());
                    let def_id = tcx.map.local_def_id(item.id);
                    let llfn = Callee::def(ccx, def_id, empty_substs).reify(ccx).val;
                    SetLinkage(llfn, linkage);
                }
            }

            item.id
        }
        FoundAst::Found(&InlinedItem::Foreign(ref item)) => {
            ccx.external().borrow_mut().insert(fn_id, Some(item.id));
            ccx.external_srcs().borrow_mut().insert(item.id, fn_id);
            item.id
        }
        FoundAst::FoundParent(parent_id, item) => {
            ccx.external().borrow_mut().insert(parent_id, Some(item.id));
            ccx.external_srcs().borrow_mut().insert(item.id, parent_id);

            let mut my_id = 0;
            match item.node {
                hir::ItemEnum(ref ast_def, _) => {
                    let ast_vs = &ast_def.variants;
                    let ty_vs = &tcx.lookup_adt_def(parent_id).variants;
                    assert_eq!(ast_vs.len(), ty_vs.len());
                    for (ast_v, ty_v) in ast_vs.iter().zip(ty_vs.iter()) {
                        if ty_v.did == fn_id { my_id = ast_v.node.data.id(); }
                        ccx.external().borrow_mut().insert(ty_v.did, Some(ast_v.node.data.id()));
                        ccx.external_srcs().borrow_mut().insert(ast_v.node.data.id(), ty_v.did);
                    }
                }
                hir::ItemStruct(ref struct_def, _) => {
                    if struct_def.is_struct() {
                        ccx.sess().bug("instantiate_inline: called on a \
                                                                 non-tuple struct")
                    } else {
                        ccx.external().borrow_mut().insert(fn_id, Some(struct_def.id()));
                        ccx.external_srcs().borrow_mut().insert(struct_def.id(), fn_id);
                        my_id = struct_def.id();
                    }
                }
                _ => ccx.sess().bug("instantiate_inline: item has a \
                                 non-enum, non-struct parent")
            }
            my_id
        }
        FoundAst::Found(&InlinedItem::TraitItem(_, ref trait_item)) => {
            ccx.external().borrow_mut().insert(fn_id, Some(trait_item.id));
            ccx.external_srcs().borrow_mut().insert(trait_item.id, fn_id);

            ccx.stats().n_inlines.set(ccx.stats().n_inlines.get() + 1);

            // Associated consts already have to be evaluated in `typeck`, so
            // the logic to do that already exists in `middle`. In order to
            // reuse that code, it needs to be able to look up the traits for
            // inlined items.
            let ty_trait_item = tcx.impl_or_trait_item(fn_id).clone();
            let trait_item_def_id = tcx.map.local_def_id(trait_item.id);
            tcx.impl_or_trait_items.borrow_mut()
               .insert(trait_item_def_id, ty_trait_item);

            // If this is a default method, we can't look up the
            // impl type. But we aren't going to translate anyways, so
            // don't.
            trait_item.id
        }
        FoundAst::Found(&InlinedItem::ImplItem(impl_did, ref impl_item)) => {
            ccx.external().borrow_mut().insert(fn_id, Some(impl_item.id));
            ccx.external_srcs().borrow_mut().insert(impl_item.id, fn_id);

            ccx.stats().n_inlines.set(ccx.stats().n_inlines.get() + 1);

            // Translate monomorphic impl methods immediately.
            if let hir::ImplItemKind::Method(ref sig, ref body) = impl_item.node {
                let impl_tpt = tcx.lookup_item_type(impl_did);
                if impl_tpt.generics.types.is_empty() &&
                        sig.generics.ty_params.is_empty() {
                    let def_id = tcx.map.local_def_id(impl_item.id);
                    let empty_substs = ccx.empty_substs_for_def_id(def_id);
                    let llfn = Callee::def(ccx, def_id, empty_substs).reify(ccx).val;
                    trans_fn(ccx,
                             &sig.decl,
                             body,
                             llfn,
                             empty_substs,
                             impl_item.id);
                    // See linkage comments on items.
                    if ccx.sess().opts.cg.codegen_units == 1 {
                        SetLinkage(llfn, InternalLinkage);
                    } else {
                        SetLinkage(llfn, AvailableExternallyLinkage);
                    }
                }
            }

            impl_item.id
        }
    };

    let inline_def_id = tcx.map.local_def_id(inline_id);
    Some(inline_def_id)
}

pub fn get_local_instance(ccx: &CrateContext, fn_id: DefId)
    -> Option<DefId> {
    if let Some(_) = ccx.tcx().map.as_local_node_id(fn_id) {
        Some(fn_id)
    } else {
        instantiate_inline(ccx, fn_id)
    }
}

pub fn maybe_instantiate_inline(ccx: &CrateContext, fn_id: DefId) -> DefId {
    get_local_instance(ccx, fn_id).unwrap_or(fn_id)
}
