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
use middle::def_id::DefId;
use middle::ty;
use middle::ty::fast_reject;
use middle::ty::{Ty, TyCtxt};
use std::borrow::{Borrow};
use std::cell::{Cell, Ref, RefCell};
use syntax::ast::Name;
use rustc_front::hir;
use util::nodemap::FnvHashMap;

/// As `TypeScheme` but for a trait ref.
pub struct TraitDef<'tcx> {
    pub unsafety: hir::Unsafety,

    /// If `true`, then this trait had the `#[rustc_paren_sugar]`
    /// attribute, indicating that it should be used with `Foo()`
    /// sugar. This is a temporary thing -- eventually any trait wil
    /// be usable with the sugar (or without it).
    pub paren_sugar: bool,

    /// Generic type definitions. Note that `Self` is listed in here
    /// as having a single bound, the trait itself (e.g., in the trait
    /// `Eq`, there is a single bound `Self : Eq`). This is so that
    /// default methods get to assume that the `Self` parameters
    /// implements the trait.
    pub generics: ty::Generics<'tcx>,

    pub trait_ref: ty::TraitRef<'tcx>,

    /// A list of the associated types defined in this trait. Useful
    /// for resolving `X::Foo` type markers.
    pub associated_type_names: Vec<Name>,

    // Impls of this trait. To allow for quicker lookup, the impls are indexed
    // by a simplified version of their Self type: impls with a simplifiable
    // Self are stored in nonblanket_impls keyed by it, while all other impls
    // are stored in blanket_impls.
    //
    // These lists are tracked by `DepNode::TraitImpls`; we don't use
    // a DepTrackingMap but instead have the `TraitDef` insert the
    // required reads/writes.

    /// Impls of the trait.
    nonblanket_impls: RefCell<
        FnvHashMap<fast_reject::SimplifiedType, Vec<DefId>>
    >,

    /// Blanket impls associated with the trait.
    blanket_impls: RefCell<Vec<DefId>>,

    /// The specialization order for impls of this trait.
    pub specialization_graph: RefCell<traits::SpecializationGraph>,

    /// Various flags
    pub flags: Cell<TraitFlags>
}

impl<'tcx> TraitDef<'tcx> {
    pub fn new(unsafety: hir::Unsafety,
               paren_sugar: bool,
               generics: ty::Generics<'tcx>,
               trait_ref: ty::TraitRef<'tcx>,
               associated_type_names: Vec<Name>)
               -> TraitDef<'tcx> {
        TraitDef {
            paren_sugar: paren_sugar,
            unsafety: unsafety,
            generics: generics,
            trait_ref: trait_ref,
            associated_type_names: associated_type_names,
            nonblanket_impls: RefCell::new(FnvHashMap()),
            blanket_impls: RefCell::new(vec![]),
            flags: Cell::new(ty::TraitFlags::NO_TRAIT_FLAGS),
            specialization_graph: RefCell::new(traits::SpecializationGraph::new()),
        }
    }

    pub fn def_id(&self) -> DefId {
        self.trait_ref.def_id
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

    fn write_trait_impls(&self, tcx: &TyCtxt<'tcx>) {
        tcx.dep_graph.write(DepNode::TraitImpls(self.trait_ref.def_id));
    }

    fn read_trait_impls(&self, tcx: &TyCtxt<'tcx>) {
        tcx.dep_graph.read(DepNode::TraitImpls(self.trait_ref.def_id));
    }

    /// Records a basic trait-to-implementation mapping.
    ///
    /// Returns `true` iff the impl has not previously been recorded.
    fn record_impl(&self,
                   tcx: &TyCtxt<'tcx>,
                   impl_def_id: DefId,
                   impl_trait_ref: TraitRef<'tcx>) -> bool {
        debug!("TraitDef::record_impl for {:?}, from {:?}",
               self, impl_trait_ref);

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
    pub fn record_local_impl(&self,
                             tcx: &TyCtxt<'tcx>,
                             impl_def_id: DefId,
                             impl_trait_ref: TraitRef<'tcx>) {
        self.record_impl(tcx, impl_def_id, impl_trait_ref);
    }

    /// Records a trait-to-implementation mapping for a non-local impl.
    ///
    /// The `parent_impl` is the immediately-less-specialized impl, or the
    /// trait's def ID if the impl is maximally-specialized -- information that
    /// should be pulled from the metadata.
    pub fn record_remote_impl(&self,
                              tcx: &TyCtxt<'tcx>,
                              impl_def_id: DefId,
                              impl_trait_ref: TraitRef<'tcx>,
                              parent_impl: DefId) {
        // if the impl has not previously been recorded
        if self.record_impl(tcx, impl_def_id, impl_trait_ref) {
            // if the impl is non-local, it's placed directly into the
            // specialization graph using parent information drawn from metadata.
            self.specialization_graph.borrow_mut()
                .record_impl_from_cstore(parent_impl, impl_def_id)
        }
    }

    /// Adds a local impl into the specialization graph, returning an error with
    /// overlap information if the impl overlaps but does not specialize an
    /// existing impl.
    pub fn add_impl_for_specialization(&self,
                                       tcx: &ctxt<'tcx>,
                                       impl_def_id: DefId,
                                       impl_trait_ref: TraitRef<'tcx>)
                                       -> Result<(), traits::Overlap<'tcx>> {
        assert!(impl_def_id.is_local());

        self.specialization_graph.borrow_mut()
            .insert(tcx, impl_def_id, impl_trait_ref)
    }

    /// Returns the immediately less specialized impl, if any.
    pub fn parent_of_impl(&self, impl_def_id: DefId) -> Option<DefId> {
        let parent = self.specialization_graph.borrow().parent(impl_def_id);
        if parent == self.trait_ref.def_id { None } else { Some(parent) }
    }

        pub fn for_each_impl<F: FnMut(DefId)>(&self, tcx: &TyCtxt<'tcx>, mut f: F)  {
            self.read_trait_impls(tcx);
        tcx.populate_implementations_for_trait_if_necessary(self.trait_ref.def_id);

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
                                                   tcx: &TyCtxt<'tcx>,
                                                   self_ty: Ty<'tcx>,
                                                   mut f: F)
    {
        self.read_trait_impls(tcx);

        tcx.populate_implementations_for_trait_if_necessary(self.trait_ref.def_id);

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

    pub fn borrow_impl_lists<'s>(&'s self, tcx: &TyCtxt<'tcx>)
                                 -> (Ref<'s, Vec<DefId>>,
                                     Ref<'s, FnvHashMap<fast_reject::SimplifiedType, Vec<DefId>>>) {
        self.read_trait_impls(tcx);
        (self.blanket_impls.borrow(), self.nonblanket_impls.borrow())
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
