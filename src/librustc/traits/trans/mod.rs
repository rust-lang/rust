// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This file contains various trait resolution methods used by trans.
// They all assume regions can be erased and monomorphic types.  It
// seems likely that they should eventually be merged into more
// general routines.

use dep_graph::{DepGraph, DepNode, DepTrackingMap, DepTrackingMapConfig};
use hir::def_id::DefId;
use infer::TransNormalize;
use std::cell::RefCell;
use std::marker::PhantomData;
use syntax::ast;
use syntax_pos::Span;
use traits::{FulfillmentContext, Obligation, ObligationCause, Reveal, SelectionContext, Vtable};
use ty::{self, Ty, TyCtxt};
use ty::subst::{Subst, Substs};
use ty::fold::{TypeFoldable, TypeFolder};
use util::common::MemoizationMap;

impl<'a, 'tcx> TyCtxt<'a, 'tcx, 'tcx> {
    /// Attempts to resolve an obligation to a vtable.. The result is
    /// a shallow vtable resolution -- meaning that we do not
    /// (necessarily) resolve all nested obligations on the impl. Note
    /// that type check should guarantee to us that all nested
    /// obligations *could be* resolved if we wanted to.
    pub fn trans_fulfill_obligation(self,
                                    span: Span,
                                    trait_ref: ty::PolyTraitRef<'tcx>)
                                    -> Vtable<'tcx, ()>
    {
        // Remove any references to regions; this helps improve caching.
        let trait_ref = self.erase_regions(&trait_ref);

        self.trans_trait_caches.trait_cache.memoize(trait_ref, || {
            debug!("trans::fulfill_obligation(trait_ref={:?}, def_id={:?})",
                   trait_ref, trait_ref.def_id());

            // Do the initial selection for the obligation. This yields the
            // shallow result we are looking for -- that is, what specific impl.
            self.infer_ctxt().enter(|infcx| {
                let mut selcx = SelectionContext::new(&infcx);

                let param_env = ty::ParamEnv::empty(Reveal::All);
                let obligation_cause = ObligationCause::misc(span,
                                                             ast::DUMMY_NODE_ID);
                let obligation = Obligation::new(obligation_cause,
                                                 param_env,
                                                 trait_ref.to_poly_trait_predicate());

                let selection = match selcx.select(&obligation) {
                    Ok(Some(selection)) => selection,
                    Ok(None) => {
                        // Ambiguity can happen when monomorphizing during trans
                        // expands to some humongo type that never occurred
                        // statically -- this humongo type can then overflow,
                        // leading to an ambiguous result. So report this as an
                        // overflow bug, since I believe this is the only case
                        // where ambiguity can result.
                        debug!("Encountered ambiguity selecting `{:?}` during trans, \
                                presuming due to overflow",
                               trait_ref);
                        self.sess.span_fatal(span,
                                            "reached the recursion limit during monomorphization \
                                             (selection ambiguity)");
                    }
                    Err(e) => {
                        span_bug!(span, "Encountered error `{:?}` selecting `{:?}` during trans",
                                  e, trait_ref)
                    }
                };

                debug!("fulfill_obligation: selection={:?}", selection);

                // Currently, we use a fulfillment context to completely resolve
                // all nested obligations. This is because they can inform the
                // inference of the impl's type parameters.
                let mut fulfill_cx = FulfillmentContext::new();
                let vtable = selection.map(|predicate| {
                    debug!("fulfill_obligation: register_predicate_obligation {:?}", predicate);
                    fulfill_cx.register_predicate_obligation(&infcx, predicate);
                });
                let vtable = infcx.drain_fulfillment_cx_or_panic(span, &mut fulfill_cx, &vtable);

                info!("Cache miss: {:?} => {:?}", trait_ref, vtable);
                vtable
            })
        })
    }

    /// Monomorphizes a type from the AST by first applying the in-scope
    /// substitutions and then normalizing any associated types.
    pub fn trans_apply_param_substs<T>(self,
                                       param_substs: &Substs<'tcx>,
                                       value: &T)
                                       -> T
        where T: TransNormalize<'tcx>
    {
        debug!("apply_param_substs(param_substs={:?}, value={:?})", param_substs, value);
        let substituted = value.subst(self, param_substs);
        let substituted = self.erase_regions(&substituted);
        AssociatedTypeNormalizer::new(self).fold(&substituted)
    }
}

struct AssociatedTypeNormalizer<'a, 'gcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'gcx>,
}

impl<'a, 'gcx> AssociatedTypeNormalizer<'a, 'gcx> {
    fn new(tcx: TyCtxt<'a, 'gcx, 'gcx>) -> Self {
        AssociatedTypeNormalizer { tcx }
    }

    fn fold<T:TypeFoldable<'gcx>>(&mut self, value: &T) -> T {
        if !value.has_projection_types() {
            value.clone()
        } else {
            value.fold_with(self)
        }
    }
}

impl<'a, 'gcx> TypeFolder<'gcx, 'gcx> for AssociatedTypeNormalizer<'a, 'gcx> {
    fn tcx<'c>(&'c self) -> TyCtxt<'c, 'gcx, 'gcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'gcx>) -> Ty<'gcx> {
        if !ty.has_projection_types() {
            ty
        } else {
            self.tcx.trans_trait_caches.project_cache.memoize(ty, || {
                debug!("AssociatedTypeNormalizer: ty={:?}", ty);
                self.tcx.normalize_associated_type(&ty)
            })
        }
    }
}

/// Specializes caches used in trans -- in particular, they assume all
/// types are fully monomorphized and that free regions can be erased.
pub struct TransTraitCaches<'tcx> {
    trait_cache: RefCell<DepTrackingMap<TraitSelectionCache<'tcx>>>,
    project_cache: RefCell<DepTrackingMap<ProjectionCache<'tcx>>>,
}

impl<'tcx> TransTraitCaches<'tcx> {
    pub fn new(graph: DepGraph) -> Self {
        TransTraitCaches {
            trait_cache: RefCell::new(DepTrackingMap::new(graph.clone())),
            project_cache: RefCell::new(DepTrackingMap::new(graph)),
        }
    }
}

// Implement DepTrackingMapConfig for `trait_cache`
pub struct TraitSelectionCache<'tcx> {
    data: PhantomData<&'tcx ()>
}

impl<'tcx> DepTrackingMapConfig for TraitSelectionCache<'tcx> {
    type Key = ty::PolyTraitRef<'tcx>;
    type Value = Vtable<'tcx, ()>;
    fn to_dep_node(key: &ty::PolyTraitRef<'tcx>) -> DepNode<DefId> {
        key.to_poly_trait_predicate().dep_node()
    }
}

// # Global Cache

pub struct ProjectionCache<'gcx> {
    data: PhantomData<&'gcx ()>
}

impl<'gcx> DepTrackingMapConfig for ProjectionCache<'gcx> {
    type Key = Ty<'gcx>;
    type Value = Ty<'gcx>;
    fn to_dep_node(key: &Self::Key) -> DepNode<DefId> {
        // Ideally, we'd just put `key` into the dep-node, but we
        // can't put full types in there. So just collect up all the
        // def-ids of structs/enums as well as any traits that we
        // project out of. It doesn't matter so much what we do here,
        // except that if we are too coarse, we'll create overly
        // coarse edges between impls and the trans. For example, if
        // we just used the def-id of things we are projecting out of,
        // then the key for `<Foo as SomeTrait>::T` and `<Bar as
        // SomeTrait>::T` would both share a dep-node
        // (`TraitSelect(SomeTrait)`), and hence the impls for both
        // `Foo` and `Bar` would be considered inputs. So a change to
        // `Bar` would affect things that just normalized `Foo`.
        // Anyway, this heuristic is not ideal, but better than
        // nothing.
        let def_ids: Vec<DefId> =
            key.walk()
               .filter_map(|t| match t.sty {
                   ty::TyAdt(adt_def, _) => Some(adt_def.did),
                   ty::TyProjection(ref proj) => Some(proj.trait_ref.def_id),
                   _ => None,
               })
               .collect();

        DepNode::ProjectionCache { def_ids: def_ids }
    }
}

