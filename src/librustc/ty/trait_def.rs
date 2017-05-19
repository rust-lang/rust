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
use ich::Fingerprint;
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
    pub def_path_hash: Fingerprint,
}

// We don't store the list of impls in a flat list because each cached list of
// `relevant_impls_for` we would then duplicate all blanket impls. By keeping
// blanket and non-blanket impls separate, we can share the list of blanket
// impls.
#[derive(Clone)]
pub struct TraitImpls {
    blanket_impls: Rc<Vec<DefId>>,
    non_blanket_impls: Rc<Vec<DefId>>,
}

impl TraitImpls {
    pub fn iter(&self) -> TraitImplsIter {
        TraitImplsIter {
            blanket_impls: self.blanket_impls.clone(),
            non_blanket_impls: self.non_blanket_impls.clone(),
            index: 0
        }
    }
}

#[derive(Clone)]
pub struct TraitImplsIter {
    blanket_impls: Rc<Vec<DefId>>,
    non_blanket_impls: Rc<Vec<DefId>>,
    index: usize,
}

impl Iterator for TraitImplsIter {
    type Item = DefId;

    fn next(&mut self) -> Option<DefId> {
        if self.index < self.blanket_impls.len() {
            let bi_index = self.index;
            self.index += 1;
            Some(self.blanket_impls[bi_index])
        } else {
            let nbi_index = self.index - self.blanket_impls.len();
            if nbi_index < self.non_blanket_impls.len() {
                self.index += 1;
                Some(self.non_blanket_impls[nbi_index])
            } else {
                None
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let items_left = (self.blanket_impls.len() + self.non_blanket_impls.len()) - self.index;
        (items_left, Some(items_left))
    }
}

impl ExactSizeIterator for TraitImplsIter {}

impl<'a, 'gcx, 'tcx> TraitDef {
    pub fn new(def_id: DefId,
               unsafety: hir::Unsafety,
               paren_sugar: bool,
               has_default_impl: bool,
               def_path_hash: Fingerprint)
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
        for impl_def_id in tcx.trait_impls_of(self.def_id).iter() {
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

        for impl_def_id in relevant_impls.iter() {
            f(impl_def_id);
        }
    }
}

// Query provider for `trait_impls_of`.
pub(super) fn trait_impls_of_provider<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                                trait_id: DefId)
                                                -> TraitImpls {
    let remote_impls = if trait_id.is_local() {
        // Traits defined in the current crate can't have impls in upstream
        // crates, so we don't bother querying the cstore.
        Vec::new()
    } else {
        tcx.sess.cstore.implementations_of_trait(Some(trait_id))
    };

    let mut blanket_impls = Vec::new();
    let mut non_blanket_impls = Vec::new();

    let local_impls = tcx.hir
                         .trait_impls(trait_id)
                         .into_iter()
                         .map(|&node_id| tcx.hir.local_def_id(node_id));

     for impl_def_id in local_impls.chain(remote_impls.into_iter()) {
        let impl_self_ty = tcx.type_of(impl_def_id);
        if impl_def_id.is_local() && impl_self_ty.references_error() {
            continue
        }

        if fast_reject::simplify_type(tcx, impl_self_ty, false).is_some() {
            non_blanket_impls.push(impl_def_id);
        } else {
            blanket_impls.push(impl_def_id);
        }
    }

    TraitImpls {
        blanket_impls: Rc::new(blanket_impls),
        non_blanket_impls: Rc::new(non_blanket_impls),
    }
}

// Query provider for `relevant_trait_impls_for`.
pub(super) fn relevant_trait_impls_provider<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    (trait_id, self_ty): (DefId, fast_reject::SimplifiedType))
    -> TraitImpls
{
    let all_trait_impls = tcx.trait_impls_of(trait_id);

    let relevant: Vec<DefId> = all_trait_impls
        .non_blanket_impls
        .iter()
        .cloned()
        .filter(|&impl_def_id| {
            let impl_self_ty = tcx.type_of(impl_def_id);
            let impl_simple_self_ty = fast_reject::simplify_type(tcx,
                                                                 impl_self_ty,
                                                                 false).unwrap();
            impl_simple_self_ty == self_ty
        })
        .collect();

    if all_trait_impls.non_blanket_impls.len() == relevant.len() {
        // If we didn't filter anything out, re-use the existing vec.
        all_trait_impls
    } else {
        TraitImpls {
            blanket_impls: all_trait_impls.blanket_impls.clone(),
            non_blanket_impls: Rc::new(relevant),
        }
    }
}
