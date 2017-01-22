// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use dep_graph::DepNode;
use hir::def_id::DefId;
use traits::{self, specialization_graph};
use ty;
use ty::fast_reject;
use ty::{Ty, TyCtxt, TraitRef};
use std::cell::{Cell, RefCell};
use hir;
use util::nodemap::FxHashMap;

/// A trait's definition with type information.
pub struct TraitDef {
    pub def_id: DefId,

    pub unsafety: hir::Unsafety,

    /// If `true`, then this trait had the `#[rustc_paren_sugar]`
    /// attribute, indicating that it should be used with `Foo()`
    /// sugar. This is a temporary thing -- eventually any trait will
    /// be usable with the sugar (or without it).
    pub paren_sugar: bool,

    // Impls of a trait. To allow for quicker lookup, the impls are indexed by a
    // simplified version of their `Self` type: impls with a simplifiable `Self`
    // are stored in `nonblanket_impls` keyed by it, while all other impls are
    // stored in `blanket_impls`.
    //
    // A similar division is used within `specialization_graph`, but the ones
    // here are (1) stored as a flat list for the trait and (2) populated prior
    // to -- and used while -- determining specialization order.
    //
    // FIXME: solve the reentrancy issues and remove these lists in favor of the
    // ones in `specialization_graph`.
    //
    // These lists are tracked by `DepNode::TraitImpls`; we don't use
    // a DepTrackingMap but instead have the `TraitDef` insert the
    // required reads/writes.

    /// Impls of the trait.
    nonblanket_impls: RefCell<
        FxHashMap<fast_reject::SimplifiedType, Vec<DefId>>
    >,

    /// Blanket impls associated with the trait.
    blanket_impls: RefCell<Vec<DefId>>,

    /// The specialization order for impls of this trait.
    pub specialization_graph: RefCell<traits::specialization_graph::Graph>,

    /// Various flags
    pub flags: Cell<TraitFlags>,

    /// The ICH of this trait's DefPath, cached here so it doesn't have to be
    /// recomputed all the time.
    pub def_path_hash: u64,
}

impl<'a, 'gcx, 'tcx> TraitDef {
    pub fn new(def_id: DefId,
               unsafety: hir::Unsafety,
               paren_sugar: bool,
               def_path_hash: u64)
               -> TraitDef {
        TraitDef {
            def_id: def_id,
            paren_sugar: paren_sugar,
            unsafety: unsafety,
            nonblanket_impls: RefCell::new(FxHashMap()),
            blanket_impls: RefCell::new(vec![]),
            flags: Cell::new(ty::TraitFlags::NO_TRAIT_FLAGS),
            specialization_graph: RefCell::new(traits::specialization_graph::Graph::new()),
            def_path_hash: def_path_hash,
        }
    }

    // returns None if not yet calculated
    pub fn object_safety(&self) -> Option<bool> {
        if self.flags.get().intersects(TraitFlags::OBJECT_SAFETY_VALID) {
            Some(self.flags.get().intersects(TraitFlags::IS_OBJECT_SAFE))
        } else {
            None
        }
    }

    pub fn set_object_safety(&self, is_safe: bool) {
        assert!(self.object_safety().map(|cs| cs == is_safe).unwrap_or(true));
        self.flags.set(
            self.flags.get() | if is_safe {
                TraitFlags::OBJECT_SAFETY_VALID | TraitFlags::IS_OBJECT_SAFE
            } else {
                TraitFlags::OBJECT_SAFETY_VALID
            }
        );
    }

    fn write_trait_impls(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) {
        tcx.dep_graph.write(DepNode::TraitImpls(self.def_id));
    }

    fn read_trait_impls(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) {
        tcx.dep_graph.read(DepNode::TraitImpls(self.def_id));
    }

    /// Records a basic trait-to-implementation mapping.
    ///
    /// Returns `true` iff the impl has not previously been recorded.
    fn record_impl(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                   impl_def_id: DefId,
                   impl_trait_ref: TraitRef<'tcx>)
                   -> bool {
        debug!("TraitDef::record_impl for {:?}, from {:?}",
               self, impl_trait_ref);

        // Record the write into the impl set, but only for local
        // impls: external impls are handled differently.
        if impl_def_id.is_local() {
            self.write_trait_impls(tcx);
        }

        // We don't want to borrow_mut after we already populated all impls,
        // so check if an impl is present with an immutable borrow first.
        if let Some(sty) = fast_reject::simplify_type(tcx,
                                                      impl_trait_ref.self_ty(), false) {
            if let Some(is) = self.nonblanket_impls.borrow().get(&sty) {
                if is.contains(&impl_def_id) {
                    return false; // duplicate - skip
                }
            }

            self.nonblanket_impls.borrow_mut().entry(sty).or_insert(vec![]).push(impl_def_id)
        } else {
            if self.blanket_impls.borrow().contains(&impl_def_id) {
                return false; // duplicate - skip
            }
            self.blanket_impls.borrow_mut().push(impl_def_id)
        }

        true
    }

    /// Records a trait-to-implementation mapping for a crate-local impl.
    pub fn record_local_impl(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                             impl_def_id: DefId,
                             impl_trait_ref: TraitRef<'tcx>) {
        assert!(impl_def_id.is_local());
        let was_new = self.record_impl(tcx, impl_def_id, impl_trait_ref);
        assert!(was_new);
    }

    /// Records a trait-to-implementation mapping for a non-local impl.
    ///
    /// The `parent_impl` is the immediately-less-specialized impl, or the
    /// trait's def ID if the impl is not a specialization -- information that
    /// should be pulled from the metadata.
    pub fn record_remote_impl(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                              impl_def_id: DefId,
                              impl_trait_ref: TraitRef<'tcx>,
                              parent_impl: DefId) {
        assert!(!impl_def_id.is_local());

        // if the impl has not previously been recorded
        if self.record_impl(tcx, impl_def_id, impl_trait_ref) {
            // if the impl is non-local, it's placed directly into the
            // specialization graph using parent information drawn from metadata.
            self.specialization_graph.borrow_mut()
                .record_impl_from_cstore(tcx, parent_impl, impl_def_id)
        }
    }

    /// Adds a local impl into the specialization graph, returning an error with
    /// overlap information if the impl overlaps but does not specialize an
    /// existing impl.
    pub fn add_impl_for_specialization(&self,
                                       tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                       impl_def_id: DefId)
                                       -> Result<(), traits::OverlapError> {
        assert!(impl_def_id.is_local());

        self.specialization_graph.borrow_mut()
            .insert(tcx, impl_def_id)
    }

    pub fn ancestors(&'a self, of_impl: DefId) -> specialization_graph::Ancestors<'a> {
        specialization_graph::ancestors(self, of_impl)
    }

    pub fn for_each_impl<F: FnMut(DefId)>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>, mut f: F) {
        self.read_trait_impls(tcx);
        tcx.populate_implementations_for_trait_if_necessary(self.def_id);

        for &impl_def_id in self.blanket_impls.borrow().iter() {
            f(impl_def_id);
        }

        for v in self.nonblanket_impls.borrow().values() {
            for &impl_def_id in v {
                f(impl_def_id);
            }
        }
    }

    /// Iterate over every impl that could possibly match the
    /// self-type `self_ty`.
    pub fn for_each_relevant_impl<F: FnMut(DefId)>(&self,
                                                   tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                                   self_ty: Ty<'tcx>,
                                                   mut f: F)
    {
        self.read_trait_impls(tcx);

        tcx.populate_implementations_for_trait_if_necessary(self.def_id);

        for &impl_def_id in self.blanket_impls.borrow().iter() {
            f(impl_def_id);
        }

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
        if let Some(simp) = fast_reject::simplify_type(tcx, self_ty, true) {
            if let Some(impls) = self.nonblanket_impls.borrow().get(&simp) {
                for &impl_def_id in impls {
                    f(impl_def_id);
                }
            }
        } else {
            for v in self.nonblanket_impls.borrow().values() {
                for &impl_def_id in v {
                    f(impl_def_id);
                }
            }
        }
    }
}

bitflags! {
    flags TraitFlags: u32 {
        const NO_TRAIT_FLAGS        = 0,
        const HAS_DEFAULT_IMPL      = 1 << 0,
        const IS_OBJECT_SAFE        = 1 << 1,
        const OBJECT_SAFETY_VALID   = 1 << 2,
        const IMPLS_VALID           = 1 << 3,
    }
}
