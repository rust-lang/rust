// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir::def_id::DefId;
use traits::specialization_graph;
use ty::fast_reject;
use ty::fold::TypeFoldable;
use ty::{Ty, TyCtxt};
use std::rc::Rc;
use hir;

/// A trait's definition with type information.
pub struct TraitDef {
    pub def_id: DefId,

    pub unsafety: hir::Unsafety,

    /// If `true`, then this trait had the `#[rustc_paren_sugar]`
    /// attribute, indicating that it should be used with `Foo()`
    /// sugar. This is a temporary thing -- eventually any trait will
    /// be usable with the sugar (or without it).
    pub paren_sugar: bool,

    pub has_default_impl: bool,

    /// The ICH of this trait's DefPath, cached here so it doesn't have to be
    /// recomputed all the time.
    pub def_path_hash: u64,
}

impl<'a, 'gcx, 'tcx> TraitDef {
    pub fn new(def_id: DefId,
               unsafety: hir::Unsafety,
               paren_sugar: bool,
               has_default_impl: bool,
               def_path_hash: u64)
               -> TraitDef {
        TraitDef {
            def_id,
            paren_sugar,
            unsafety,
            has_default_impl,
            def_path_hash,
        }
    }

    pub fn ancestors(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                     of_impl: DefId)
                     -> specialization_graph::Ancestors {
        specialization_graph::ancestors(tcx, self.def_id, of_impl)
    }

    pub fn for_each_impl<F: FnMut(DefId)>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>, mut f: F) {
        for &impl_def_id in tcx.trait_impls_of(self.def_id).iter() {
            f(impl_def_id);
        }
    }

    /// Iterate over every impl that could possibly match the
    /// self-type `self_ty`.
    pub fn for_each_relevant_impl<F: FnMut(DefId)>(&self,
                                                   tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                                   self_ty: Ty<'tcx>,
                                                   mut f: F)
    {
        // simplify_type(.., false) basically replaces type parameters and
        // projections with infer-variables. This is, of course, done on
        // the impl trait-ref when it is instantiated, but not on the
        // predicate trait-ref which is passed here.
        //
        // for example, if we match `S: Copy` against an impl like
        // `impl<T:Copy> Copy for Option<T>`, we replace the type variable
        // in `Option<T>` with an infer variable, to `Option<_>` (this
        // doesn't actually change fast_reject output), but we don't
        // replace `S` with anything - this impl of course can't be
        // selected, and as there are hundreds of similar impls,
        // considering them would significantly harm performance.
        let relevant_impls = if let Some(simplified_self_ty) =
                fast_reject::simplify_type(tcx, self_ty, true) {
            tcx.relevant_trait_impls_for((self.def_id, simplified_self_ty))
        } else {
            tcx.trait_impls_of(self.def_id)
        };

        for &impl_def_id in relevant_impls.iter() {
            f(impl_def_id);
        }
    }
}

// Query provider for `trait_impls_of`.
pub(super) fn trait_impls_of_provider<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                                trait_id: DefId)
                                                -> Rc<Vec<DefId>> {
    let mut impls = if trait_id.is_local() {
        // Traits defined in the current crate can't have impls in upstream
        // crates, so we don't bother querying the cstore.
        Vec::new()
    } else {
        tcx.sess.cstore.implementations_of_trait(Some(trait_id))
    };

    impls.extend(tcx.hir
                    .trait_impls(trait_id)
                    .iter()
                    .map(|&node_id| tcx.hir.local_def_id(node_id))
                    .filter(|&impl_def_id| {
                        let trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap();
                        !trait_ref.references_error()
                    }));
    Rc::new(impls)
}

// Query provider for `relevant_trait_impls_for`.
pub(super) fn relevant_trait_impls_provider<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    (trait_id, self_ty): (DefId, fast_reject::SimplifiedType))
    -> Rc<Vec<DefId>>
{
    let all_trait_impls = tcx.trait_impls_of(trait_id);

    let relevant: Vec<DefId> = all_trait_impls
        .iter()
        .map(|&impl_def_id| impl_def_id)
        .filter(|&impl_def_id| {
            let impl_trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap();
            let impl_simple_self_ty = fast_reject::simplify_type(tcx,
                                                                 impl_trait_ref.self_ty(),
                                                                 false);
            if let Some(impl_simple_self_ty) = impl_simple_self_ty {
                impl_simple_self_ty == self_ty
            } else {
                // blanket impl (?)
                true
            }
        })
        .collect();

    if all_trait_impls.len() == relevant.len() {
        // If we didn't filter anything out, re-use the existing vec.
        all_trait_impls
    } else {
        Rc::new(relevant)
    }
}
