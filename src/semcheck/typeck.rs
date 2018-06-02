//! The type and predicate checking logic used to compare types of corresponding items.
//!
//! Multiple context structures are provided that modularize the needed functionality to allow
//! for code reuse across analysis steps.

use rustc::hir::def_id::DefId;
use rustc::infer::InferCtxt;
use rustc::traits::{FulfillmentContext, FulfillmentError,
                    Obligation, ObligationCause, TraitEngine};
use rustc::ty::{GenericParamDefKind, ParamEnv, Predicate, TraitRef, Ty, TyCtxt};
use rustc::ty::error::TypeError;
use rustc::ty::fold::TypeFoldable;
use rustc::ty::subst::{Kind, Substs};

use semcheck::changes::ChangeSet;
use semcheck::mapping::IdMapping;
use semcheck::translate::{InferenceCleanupFolder, TranslationContext};

/// The context in which bounds analysis happens.
pub struct BoundContext<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    /// The inference context to use.
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
    /// The fulfillment context to use.
    fulfill_cx: FulfillmentContext<'tcx>,
    /// The param env to be assumed.
    given_param_env: ParamEnv<'tcx>,
}

impl<'a, 'gcx, 'tcx> BoundContext<'a, 'gcx, 'tcx> {
    /// Construct a new bound context.
    pub fn new(infcx: &'a InferCtxt<'a, 'gcx, 'tcx>, given_param_env: ParamEnv<'tcx>) -> Self {
        BoundContext {
            infcx: infcx,
            fulfill_cx: FulfillmentContext::new(),
            given_param_env: given_param_env,
        }
    }

    /// Register the bounds of an item.
    pub fn register(&mut self, checked_def_id: DefId, substs: &Substs<'tcx>) {
        use rustc::traits::{normalize, Normalized, SelectionContext};

        let cause = ObligationCause::dummy();
        let mut selcx = SelectionContext::new(self.infcx);
        let predicates =
            self.infcx
                .tcx
                .predicates_of(checked_def_id)
                .instantiate(self.infcx.tcx, substs);
        let Normalized { value, obligations } =
            normalize(&mut selcx, self.given_param_env, cause.clone(), &predicates);

        for obligation in obligations {
            self.fulfill_cx.register_predicate_obligation(self.infcx, obligation);
        }

        for predicate in value.predicates {
            let obligation = Obligation::new(cause.clone(), self.given_param_env, predicate);
            self.fulfill_cx.register_predicate_obligation(self.infcx, obligation);
        }
    }

    /// Register the trait bound represented by a `TraitRef`.
    pub fn register_trait_ref(&mut self, checked_trait_ref: TraitRef<'tcx>) {
        use rustc::ty::{Binder, Predicate, TraitPredicate};

        let predicate = Predicate::Trait(Binder::bind(TraitPredicate {
            trait_ref: checked_trait_ref,
        }));
        let obligation =
            Obligation::new(ObligationCause::dummy(), self.given_param_env, predicate);
        self.fulfill_cx.register_predicate_obligation(self.infcx, obligation);
    }

    /// Return inference errors, if any.
    pub fn get_errors(&mut self) -> Option<Vec<FulfillmentError<'tcx>>> {
        if let Err(err) = self.fulfill_cx.select_all_or_error(self.infcx) {
            debug!("err: {:?}", err);
            Some(err)
        } else {
            None
        }
    }
}

/// The context in which types and their bounds can be compared.
pub struct TypeComparisonContext<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    /// The inference context to use.
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
    /// The index mapping to use.
    id_mapping: &'a IdMapping,
    /// The folder to clean up found errors of inference artifacts.
    folder: InferenceCleanupFolder<'a, 'gcx, 'tcx>,
    /// The translation context translating from original to target items.
    pub forward_trans: TranslationContext<'a, 'gcx, 'tcx>,
    /// The translation context translating from target to original items.
    pub backward_trans: TranslationContext<'a, 'gcx, 'tcx>,
    /// Whether we are checking a trait definition.
    checking_trait_def: bool,
}

impl<'a, 'gcx, 'tcx> TypeComparisonContext<'a, 'gcx, 'tcx> {
    /// Construct a new context where the original item is old.
    pub fn target_new(infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
                      id_mapping: &'a IdMapping,
                      checking_trait_def: bool) -> Self {
        let forward_trans = TranslationContext::target_new(infcx.tcx, id_mapping, false);
        let backward_trans = TranslationContext::target_old(infcx.tcx, id_mapping, false);
        TypeComparisonContext::from_trans(infcx,
                                          id_mapping,
                                          forward_trans,
                                          backward_trans,
                                          checking_trait_def)
    }

    /// Construct a new context where the original item is new.
    pub fn target_old(infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
                      id_mapping: &'a IdMapping,
                      checking_trait_def: bool) -> Self {
        let forward_trans = TranslationContext::target_old(infcx.tcx, id_mapping, false);
        let backward_trans = TranslationContext::target_new(infcx.tcx, id_mapping, false);
        TypeComparisonContext::from_trans(infcx,
                                          id_mapping,
                                          forward_trans,
                                          backward_trans,
                                          checking_trait_def)
    }

    /// Construct a new context given a pair of translation contexts.
    fn from_trans(infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
                  id_mapping: &'a IdMapping,
                  forward_trans: TranslationContext<'a, 'gcx, 'tcx>,
                  backward_trans: TranslationContext<'a, 'gcx, 'tcx>,
                  checking_trait_def: bool) -> Self {
        TypeComparisonContext {
            infcx: infcx,
            id_mapping: id_mapping,
            folder: InferenceCleanupFolder::new(infcx),
            forward_trans: forward_trans,
            backward_trans: backward_trans,
            checking_trait_def: checking_trait_def,
        }
    }

    /// Construct a set of subsitutions for an item, which replaces all region and type variables
    /// with inference variables, with the exception of `Self`.
    pub fn compute_target_infer_substs(&self, target_def_id: DefId) -> &Substs<'tcx> {
        use syntax_pos::DUMMY_SP;

        let has_self = self.infcx.tcx.generics_of(target_def_id).has_self;

        Substs::for_item(self.infcx.tcx, target_def_id, |def, _| {
            if def.index == 0 && has_self { // `Self` is special
                self.infcx.tcx.mk_param_from_def(def)
            } else {
                self.infcx.var_for_def(DUMMY_SP, def)
            }
        })
    }

    /// Construct a set of subsitutions for an item, which normalizes defaults.
    pub fn compute_target_default_substs(&self, target_def_id: DefId) -> &Substs<'tcx> {
        use rustc::ty::ReEarlyBound;

        Substs::for_item(self.infcx.tcx, target_def_id, |def, _| {
            match def.kind {
                GenericParamDefKind::Lifetime => {
                    Kind::from(
                        self.infcx
                            .tcx
                            .mk_region(ReEarlyBound(def.to_early_bound_region_data())))
                },
                GenericParamDefKind::Type { .. } => {
                    if self.id_mapping.is_non_mapped_defaulted_type_param(&def.def_id) {
                        Kind::from(self.infcx.tcx.type_of(def.def_id))
                    } else {
                        self.infcx.tcx.mk_param_from_def(def)
                    }
                },
            }
        })
    }

    /// Check for type mismatches in a pair of items.
    pub fn check_type_error<'b, 'tcx2>(&self,
                                       lift_tcx: TyCtxt<'b, 'tcx2, 'tcx2>,
                                       target_def_id: DefId,
                                       target_param_env: ParamEnv<'tcx>,
                                       orig: Ty<'tcx>,
                                       target: Ty<'tcx>) -> Option<TypeError<'tcx2>> {
        use rustc::infer::InferOk;
        use rustc::infer::outlives::env::OutlivesEnvironment;
        use rustc::middle::region::ScopeTree;
        use rustc::ty::Lift;

        let error =
            self.infcx
                .at(&ObligationCause::dummy(), target_param_env)
                .eq(orig, target)
                .map(|InferOk { obligations: o, .. }| { assert_eq!(o, vec![]); });

        if let Err(err) = error {
            let scope_tree = ScopeTree::default();
            let mut outlives_env = OutlivesEnvironment::new(target_param_env);

            // The old code here added the bounds from the target param env by hand. However, at
            // least the explicit bounds are added when the OutlivesEnvironment is created. This
            // seems to work, but in case it stops to do so, the below code snippets should be
            // of help to implement the old behaviour.
            //
            // outlives_env.add_outlives_bounds(None, target_param_env.caller_bounds.iter()....)
            // free_regions.relate_free_regions_from_predicates(target_param_env.caller_bounds);
            //  ty::Predicate::RegionOutlives(ty::Binder(ty::OutlivesPredicate(r_a, r_b))) => {
            //      self.relate_regions(r_b, r_a);
            //  }

            self.infcx.resolve_regions_and_report_errors(target_def_id,
                                                         &scope_tree,
                                                         &outlives_env);

            let err =
                self.infcx
                    .resolve_type_vars_if_possible(&err)
                    .fold_with(&mut self.folder.clone())
                    .lift_to_tcx(lift_tcx)
                    .unwrap();

            Some(err)
        } else {
            None
        }
    }

    /// Check for trait bound mismatches in a pair of items.
    pub fn check_bounds_error<'b, 'tcx2>(&self,
                                         lift_tcx: TyCtxt<'b, 'tcx2, 'tcx2>,
                                         orig_param_env: ParamEnv<'tcx>,
                                         target_def_id: DefId,
                                         target_substs: &Substs<'tcx>)
        -> Option<Vec<Predicate<'tcx2>>>
    {
        use rustc::ty::Lift;
        debug!("check_bounds_error: orig env: {:?}, target did: {:?}, target substs: {:?}",
               orig_param_env,
               target_def_id,
               target_substs);

        let mut bound_cx = BoundContext::new(self.infcx, orig_param_env);
        bound_cx.register(target_def_id, target_substs);

        bound_cx
            .get_errors()
            .map(|errors| errors
                .iter()
                .map(|err|
                     self.infcx
                         .resolve_type_vars_if_possible(&err.obligation.predicate)
                         .fold_with(&mut self.folder.clone())
                         .lift_to_tcx(lift_tcx)
                         .unwrap())
                .collect())
    }

    /// Check the bounds on an item in both directions and register changes found.
    pub fn check_bounds_bidirectional<'b, 'tcx2>(&self,
                                                 changes: &mut ChangeSet<'tcx2>,
                                                 lift_tcx: TyCtxt<'b, 'tcx2, 'tcx2>,
                                                 orig_def_id: DefId,
                                                 target_def_id: DefId,
                                                 orig_substs: &Substs<'tcx>,
                                                 target_substs: &Substs<'tcx>) {
        use semcheck::changes::ChangeType::{BoundsLoosened, BoundsTightened};

        let tcx = self.infcx.tcx;

        let orig_param_env = self
            .forward_trans
            .translate_param_env(orig_def_id, tcx.param_env(orig_def_id));

        let orig_param_env = if let Some(env) = orig_param_env {
            env
        } else {
            return;
        };

        if let Some(errors) =
            self.check_bounds_error(lift_tcx, orig_param_env, target_def_id, target_substs)
        {
            for err in errors {
                let err_type = BoundsTightened {
                    pred: err,
                };

                changes.add_change(err_type, orig_def_id, None);
            }
        }

        let target_param_env = self
            .backward_trans
            .translate_param_env(target_def_id, tcx.param_env(target_def_id));

        let target_param_env = if let Some(env) = target_param_env {
            env
        } else {
            return;
        };

        if let Some(errors) =
            self.check_bounds_error(lift_tcx, target_param_env, orig_def_id, orig_substs)
        {
            for err in errors {
                let err_type = BoundsLoosened {
                    pred: err,
                    trait_def: self.checking_trait_def,
                };

                changes.add_change(err_type, orig_def_id, None);
            }
        }
    }
}
