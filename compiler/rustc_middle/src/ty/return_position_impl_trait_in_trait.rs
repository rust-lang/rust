use rustc_hir::def_id::DefId;

use crate::ty::{self, ExistentialPredicateStableCmpExt, TyCtxt};

impl<'tcx> TyCtxt<'tcx> {
    /// Given a `def_id` of a trait or impl method, compute whether that method needs to
    /// have an RPITIT shim applied to it for it to be dyn compatible. If so, return the
    /// `def_id` of the RPITIT, and also the args of trait method that returns the RPITIT.
    ///
    /// NOTE that these args are not, in general, the same as than the RPITIT's args. They
    /// are a subset of those args,  since they do not include the late-bound lifetimes of
    /// the RPITIT. Depending on the context, these will need to be dealt with in different
    /// ways -- in codegen, it's okay to fill them with ReErased.
    pub fn return_position_impl_trait_in_trait_shim_data(
        self,
        def_id: DefId,
    ) -> Option<(DefId, ty::EarlyBinder<'tcx, ty::GenericArgsRef<'tcx>>)> {
        let assoc_item = self.opt_associated_item(def_id)?;

        let (trait_item_def_id, opt_impl_def_id) = match assoc_item.container {
            ty::AssocItemContainer::Impl => {
                (assoc_item.trait_item_def_id?, Some(self.parent(def_id)))
            }
            ty::AssocItemContainer::Trait => (def_id, None),
        };

        let sig = self.fn_sig(trait_item_def_id);

        // Check if the trait returns an RPITIT.
        let ty::Alias(ty::Projection, ty::AliasTy { def_id, .. }) =
            *sig.skip_binder().skip_binder().output().kind()
        else {
            return None;
        };
        if !self.is_impl_trait_in_trait(def_id) {
            return None;
        }

        let args = if let Some(impl_def_id) = opt_impl_def_id {
            // Rebase the args from the RPITIT onto the impl trait ref, so we can later
            // substitute them with the method args of the *impl* method, since that's
            // the instance we're building a vtable shim for.
            ty::GenericArgs::identity_for_item(self, trait_item_def_id).rebase_onto(
                self,
                self.parent(trait_item_def_id),
                self.impl_trait_ref(impl_def_id)
                    .expect("expected impl trait ref from parent of impl item")
                    .instantiate_identity()
                    .args,
            )
        } else {
            // This is when we have a default trait implementation.
            ty::GenericArgs::identity_for_item(self, trait_item_def_id)
        };

        Some((def_id, ty::EarlyBinder::bind(args)))
    }

    /// Given a `DefId` of an RPITIT and its args, return the existential predicates
    /// that corresponds to the RPITIT's bounds with the self type erased.
    pub fn item_bounds_to_existential_predicates(
        self,
        def_id: DefId,
        args: ty::GenericArgsRef<'tcx>,
    ) -> &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>> {
        let mut bounds: Vec<_> = self
            .item_self_bounds(def_id)
            .iter_instantiated(self, args)
            .filter_map(|clause| {
                clause
                    .kind()
                    .map_bound(|clause| match clause {
                        ty::ClauseKind::Trait(trait_pred) => Some(ty::ExistentialPredicate::Trait(
                            ty::ExistentialTraitRef::erase_self_ty(self, trait_pred.trait_ref),
                        )),
                        ty::ClauseKind::Projection(projection_pred) => {
                            Some(ty::ExistentialPredicate::Projection(
                                ty::ExistentialProjection::erase_self_ty(self, projection_pred),
                            ))
                        }
                        ty::ClauseKind::TypeOutlives(_) => {
                            // Type outlives bounds don't really turn into anything,
                            // since we must use an intersection region for the `dyn*`'s
                            // region anyways.
                            None
                        }
                        _ => unreachable!("unexpected clause in item bounds: {clause:?}"),
                    })
                    .transpose()
            })
            .collect();
        bounds.sort_by(|a, b| a.skip_binder().stable_cmp(self, &b.skip_binder()));
        self.mk_poly_existential_predicates(&bounds)
    }
}
