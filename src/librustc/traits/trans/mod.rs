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

use dep_graph::{DepKind, DepTrackingMapConfig};
use infer::TransNormalize;
use std::marker::PhantomData;
use syntax_pos::DUMMY_SP;
use hir::def_id::DefId;
use traits::{FulfillmentContext, Obligation, ObligationCause, SelectionContext, Vtable};
use ty::{self, Ty, TyCtxt};
use ty::subst::{Subst, Substs};
use ty::fold::{TypeFoldable, TypeFolder};

/// Attempts to resolve an obligation to a vtable.. The result is
/// a shallow vtable resolution -- meaning that we do not
/// (necessarily) resolve all nested obligations on the impl. Note
/// that type check should guarantee to us that all nested
/// obligations *could be* resolved if we wanted to.
/// Assumes that this is run after the entire crate has been successfully type-checked.
pub fn trans_fulfill_obligation<'a, 'tcx>(ty: TyCtxt<'a, 'tcx, 'tcx>,
                                          (param_env, trait_ref):
                                          (ty::ParamEnv<'tcx>, ty::PolyTraitRef<'tcx>))
                                          -> Vtable<'tcx, ()>
{
    // Remove any references to regions; this helps improve caching.
    let trait_ref = ty.erase_regions(&trait_ref);

    debug!("trans::fulfill_obligation(trait_ref={:?}, def_id={:?})",
            (param_env, trait_ref), trait_ref.def_id());

    // Do the initial selection for the obligation. This yields the
    // shallow result we are looking for -- that is, what specific impl.
    ty.infer_ctxt().enter(|infcx| {
        let mut selcx = SelectionContext::new(&infcx);

        let obligation_cause = ObligationCause::dummy();
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
                bug!("Encountered ambiguity selecting `{:?}` during trans, \
                        presuming due to overflow",
                        trait_ref)
            }
            Err(e) => {
                bug!("Encountered error `{:?}` selecting `{:?}` during trans",
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
        let vtable = infcx.drain_fulfillment_cx_or_panic(DUMMY_SP, &mut fulfill_cx, &vtable);

        info!("Cache miss: {:?} => {:?}", trait_ref, vtable);
        vtable
    })
}

impl<'a, 'tcx> TyCtxt<'a, 'tcx, 'tcx> {
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

    pub fn trans_apply_param_substs_env<T>(
        self,
        param_substs: &Substs<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        value: &T,
    ) -> T
    where
        T: TransNormalize<'tcx>,
    {
        debug!(
            "apply_param_substs_env(param_substs={:?}, value={:?}, param_env={:?})",
            param_substs,
            value,
            param_env,
        );
        let substituted = value.subst(self, param_substs);
        let substituted = self.erase_regions(&substituted);
        AssociatedTypeNormalizerEnv::new(self, param_env).fold(&substituted)
    }

    pub fn trans_impl_self_ty(&self, def_id: DefId, substs: &'tcx Substs<'tcx>)
                              -> Ty<'tcx>
    {
        self.trans_apply_param_substs(substs, &self.type_of(def_id))
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
        if !value.has_projections() {
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
        if !ty.has_projections() {
            ty
        } else {
            debug!("AssociatedTypeNormalizer: ty={:?}", ty);
            self.tcx.fully_normalize_monormophic_ty(ty)
        }
    }
}

struct AssociatedTypeNormalizerEnv<'a, 'gcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'gcx>,
    param_env: ty::ParamEnv<'gcx>,
}

impl<'a, 'gcx> AssociatedTypeNormalizerEnv<'a, 'gcx> {
    fn new(tcx: TyCtxt<'a, 'gcx, 'gcx>, param_env: ty::ParamEnv<'gcx>) -> Self {
        Self { tcx, param_env }
    }

    fn fold<T: TypeFoldable<'gcx>>(&mut self, value: &T) -> T {
        if !value.has_projections() {
            value.clone()
        } else {
            value.fold_with(self)
        }
    }
}

impl<'a, 'gcx> TypeFolder<'gcx, 'gcx> for AssociatedTypeNormalizerEnv<'a, 'gcx> {
    fn tcx<'c>(&'c self) -> TyCtxt<'c, 'gcx, 'gcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'gcx>) -> Ty<'gcx> {
        if !ty.has_projections() {
            ty
        } else {
            debug!("AssociatedTypeNormalizerEnv: ty={:?}", ty);
            self.tcx.normalize_associated_type_in_env(&ty, self.param_env)
        }
    }
}

// Implement DepTrackingMapConfig for `trait_cache`
pub struct TraitSelectionCache<'tcx> {
    data: PhantomData<&'tcx ()>
}

impl<'tcx> DepTrackingMapConfig for TraitSelectionCache<'tcx> {
    type Key = (ty::ParamEnv<'tcx>, ty::PolyTraitRef<'tcx>);
    type Value = Vtable<'tcx, ()>;
    fn to_dep_kind() -> DepKind {
        DepKind::TraitSelect
    }
}

// # Global Cache

pub struct ProjectionCache<'gcx> {
    data: PhantomData<&'gcx ()>
}

impl<'gcx> DepTrackingMapConfig for ProjectionCache<'gcx> {
    type Key = Ty<'gcx>;
    type Value = Ty<'gcx>;
    fn to_dep_kind() -> DepKind {
        DepKind::TraitSelect
    }
}
