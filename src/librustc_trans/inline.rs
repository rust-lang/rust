// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::def_id::DefId;
use base::push_ctxt;
use common::*;
use monomorphize::Instance;

use rustc::dep_graph::DepNode;

fn instantiate_inline(ccx: &CrateContext, fn_id: DefId) -> Option<DefId> {
    debug!("instantiate_inline({:?})", fn_id);
    let _icx = push_ctxt("instantiate_inline");
    let tcx = ccx.tcx();
    let _task = tcx.dep_graph.in_task(DepNode::TransInlinedItem(fn_id));

    tcx.sess
       .cstore
       .maybe_get_item_ast(tcx, fn_id)
       .map(|(_, inline_id)| {
            tcx.map.local_def_id(inline_id)
       })
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

pub fn maybe_inline_instance<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                       instance: Instance<'tcx>) -> Instance<'tcx> {
    let def_id = maybe_instantiate_inline(ccx, instance.def);
    Instance {
        def: def_id,
        substs: instance.substs
    }
}
