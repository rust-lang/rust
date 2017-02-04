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

use dep_graph::DepTrackingMap;
use hir::def_id::DefId;
use rustc::ty::{self, maps, TyCtxt, TypeFoldable};
use rustc::ty::{Ty, TyBool, TyChar, TyError};
use rustc::ty::{TyParam, TyRawPtr};
use rustc::ty::{TyRef, TyAdt, TyDynamic, TyNever, TyTuple};
use rustc::ty::{TyStr, TyArray, TySlice, TyFloat, TyInfer, TyInt};
use rustc::ty::{TyUint, TyClosure, TyFnDef, TyFnPtr};
use rustc::ty::{TyProjection, TyAnon};
use CrateCtxt;
use syntax_pos::Span;
use rustc::dep_graph::DepNode;
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::hir::{Item, ItemImpl};
use rustc::hir;
use std::cell::RefMut;

mod builtin;
mod orphan;
mod overlap;
mod unsafety;

struct CoherenceCollect<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    inherent_impls: RefMut<'a, DepTrackingMap<maps::InherentImpls<'tcx>>>,
}

impl<'a, 'tcx, 'v> ItemLikeVisitor<'v> for CoherenceCollect<'a, 'tcx> {
    fn visit_item(&mut self, item: &Item) {
        if let ItemImpl(..) = item.node {
            self.check_implementation(item)
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem) {
    }

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem) {
    }
}

impl<'a, 'tcx> CoherenceCollect<'a, 'tcx> {
    fn check(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
        let inherent_impls = tcx.inherent_impls.borrow_mut();
        let mut this = &mut CoherenceCollect { tcx, inherent_impls };

        // Check implementations and traits. This populates the tables
        // containing the inherent methods and extension methods. It also
        // builds up the trait inheritance table.
        tcx.visit_all_item_likes_in_krate(DepNode::CoherenceCheckImpl, this);
    }

    // Returns the def ID of the base type, if there is one.
    fn get_base_type_def_id(&self, span: Span, ty: Ty<'tcx>) -> Option<DefId> {
        match ty.sty {
            TyAdt(def, _) => Some(def.did),

            TyDynamic(ref t, ..) => t.principal().map(|p| p.def_id()),

            TyBool | TyChar | TyInt(..) | TyUint(..) | TyFloat(..) | TyStr | TyArray(..) |
            TySlice(..) | TyFnDef(..) | TyFnPtr(_) | TyTuple(..) | TyParam(..) | TyError |
            TyNever | TyRawPtr(_) | TyRef(..) | TyProjection(..) => None,

            TyInfer(..) | TyClosure(..) | TyAnon(..) => {
                // `ty` comes from a user declaration so we should only expect types
                // that the user can type
                span_bug!(span,
                          "coherence encountered unexpected type searching for base type: {}",
                          ty);
            }
        }
    }

    fn check_implementation(&mut self, item: &Item) {
        let tcx = self.tcx;
        let impl_did = tcx.hir.local_def_id(item.id);
        let self_type = tcx.item_type(impl_did);

        // If there are no traits, then this implementation must have a
        // base type.

        if let Some(trait_ref) = self.tcx.impl_trait_ref(impl_did) {
            debug!("(checking implementation) adding impl for trait '{:?}', item '{}'",
                   trait_ref,
                   item.name);

            // Skip impls where one of the self type is an error type.
            // This occurs with e.g. resolve failures (#30589).
            if trait_ref.references_error() {
                return;
            }

            enforce_trait_manually_implementable(self.tcx, item.span, trait_ref.def_id);
            self.add_trait_impl(trait_ref, impl_did);
        } else {
            // Skip inherent impls where the self type is an error
            // type. This occurs with e.g. resolve failures (#30589).
            if self_type.references_error() {
                return;
            }

            // Add the implementation to the mapping from implementation to base
            // type def ID, if there is a base type for this implementation and
            // the implementation does not have any associated traits.
            if let Some(base_def_id) = self.get_base_type_def_id(item.span, self_type) {
                self.add_inherent_impl(base_def_id, impl_did);
            }
        }
    }

    fn add_inherent_impl(&mut self, base_def_id: DefId, impl_def_id: DefId) {
        // Subtle: it'd be better to collect these into a local map
        // and then write the vector only once all items are known,
        // but that leads to degenerate dep-graphs. The problem is
        // that the write of that big vector winds up having reads
        // from *all* impls in the krate, since we've lost the
        // precision basically.  This would be ok in the firewall
        // model so once we've made progess towards that we can modify
        // the strategy here. In the meantime, using `push` is ok
        // because we are doing this as a pre-pass before anyone
        // actually reads from `inherent_impls` -- and we know this is
        // true beacuse we hold the refcell lock.
        self.inherent_impls.push(base_def_id, impl_def_id);
    }

    fn add_trait_impl(&self, impl_trait_ref: ty::TraitRef<'tcx>, impl_def_id: DefId) {
        debug!("add_trait_impl: impl_trait_ref={:?} impl_def_id={:?}",
               impl_trait_ref,
               impl_def_id);
        let trait_def = self.tcx.lookup_trait_def(impl_trait_ref.def_id);
        trait_def.record_local_impl(self.tcx, impl_def_id, impl_trait_ref);
    }
}

fn enforce_trait_manually_implementable(tcx: TyCtxt, sp: Span, trait_def_id: DefId) {
    if tcx.sess.features.borrow().unboxed_closures {
        // the feature gate allows all of them
        return;
    }
    let did = Some(trait_def_id);
    let li = &tcx.lang_items;

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
                                   sp,
                                   E0183,
                                   "manual implementations of `{}` are experimental",
                                   trait_name);
    help!(&mut err,
          "add `#![feature(unboxed_closures)]` to the crate attributes to enable");
    err.emit();
}

pub fn check_coherence(ccx: &CrateCtxt) {
    CoherenceCollect::check(ccx.tcx);

    let _task = ccx.tcx.dep_graph.in_task(DepNode::Coherence);
    unsafety::check(ccx.tcx);
    orphan::check(ccx.tcx);
    overlap::check(ccx.tcx);
    builtin::check(ccx.tcx);
}
