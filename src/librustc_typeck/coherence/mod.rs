// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Coherence phase
//
// The job of the coherence phase of typechecking is to ensure that
// each trait has at most one implementation for each type. This is
// done by the orphan and overlap modules. Then we build up various
// mappings. That mapping code resides here.

use hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc::ty::{self, TyCtxt, TypeFoldable};
use rustc::ty::maps::Providers;
use rustc::dep_graph::DepNode;

use syntax::ast;
use syntax_pos::DUMMY_SP;

mod builtin;
mod inherent;
mod orphan;
mod overlap;
mod unsafety;

fn check_impl<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, node_id: ast::NodeId) {
    let impl_def_id = tcx.hir.local_def_id(node_id);

    // If there are no traits, then this implementation must have a
    // base type.

    if let Some(trait_ref) = tcx.impl_trait_ref(impl_def_id) {
        debug!("(checking implementation) adding impl for trait '{:?}', item '{}'",
                trait_ref,
                tcx.item_path_str(impl_def_id));

        // Skip impls where one of the self type is an error type.
        // This occurs with e.g. resolve failures (#30589).
        if trait_ref.references_error() {
            return;
        }

        enforce_trait_manually_implementable(tcx, impl_def_id, trait_ref.def_id);
        let trait_def = tcx.lookup_trait_def(trait_ref.def_id);
        trait_def.record_local_impl(tcx, impl_def_id, trait_ref);
    }
}

fn enforce_trait_manually_implementable(tcx: TyCtxt, impl_def_id: DefId, trait_def_id: DefId) {
    let did = Some(trait_def_id);
    let li = &tcx.lang_items;

    // Disallow *all* explicit impls of `Sized` and `Unsize` for now.
    if did == li.sized_trait() {
        let span = tcx.span_of_impl(impl_def_id).unwrap();
        struct_span_err!(tcx.sess,
                         span,
                         E0322,
                         "explicit impls for the `Sized` trait are not permitted")
            .span_label(span, &format!("impl of 'Sized' not allowed"))
            .emit();
        return;
    }

    if did == li.unsize_trait() {
        let span = tcx.span_of_impl(impl_def_id).unwrap();
        span_err!(tcx.sess,
                  span,
                  E0328,
                  "explicit impls for the `Unsize` trait are not permitted");
        return;
    }

    if tcx.sess.features.borrow().unboxed_closures {
        // the feature gate allows all Fn traits
        return;
    }

    let trait_name = if did == li.fn_trait() {
        "Fn"
    } else if did == li.fn_mut_trait() {
        "FnMut"
    } else if did == li.fn_once_trait() {
        "FnOnce"
    } else {
        return; // everything OK
    };
    let mut err = struct_span_err!(tcx.sess,
                                   tcx.span_of_impl(impl_def_id).unwrap(),
                                   E0183,
                                   "manual implementations of `{}` are experimental",
                                   trait_name);
    help!(&mut err,
          "add `#![feature(unboxed_closures)]` to the crate attributes to enable");
    err.emit();
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        coherent_trait,
        coherent_inherent_impls,
        ..*providers
    };
}

fn coherent_trait<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            (_, def_id): (CrateNum, DefId)) {
    tcx.populate_implementations_for_trait_if_necessary(def_id);

    let impls = tcx.hir.trait_impls(def_id);
    for &impl_id in impls {
        check_impl(tcx, impl_id);
    }
    for &impl_id in impls {
        overlap::check_impl(tcx, impl_id);
    }
    builtin::check_trait(tcx, def_id);
}

fn coherent_inherent_impls<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, _: CrateNum) {
    inherent::check(tcx);
}

pub fn check_coherence<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let _task = tcx.dep_graph.in_task(DepNode::Coherence);
    for &trait_def_id in tcx.hir.krate().trait_impls.keys() {
        ty::queries::coherent_trait::get(tcx, DUMMY_SP, (LOCAL_CRATE, trait_def_id));
    }

    unsafety::check(tcx);
    orphan::check(tcx);
    overlap::check_default_impls(tcx);

    ty::queries::coherent_inherent_impls::get(tcx, DUMMY_SP, LOCAL_CRATE);
}
