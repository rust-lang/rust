// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty::TypeFoldable;
use std::fmt::Debug;

use super::*;

pub struct AutoTraitFinder<'a, 'tcx: 'a, 'rcx: 'a> {
    pub cx: &'a core::DocContext<'a, 'tcx, 'rcx>,
}

impl<'a, 'tcx, 'rcx> AutoTraitFinder<'a, 'tcx, 'rcx> {
    pub fn get_with_def_id(&self, def_id: DefId) -> Vec<Item> {
        let ty = self.cx.tcx.type_of(def_id);

        let def_ctor: fn(DefId) -> Def = match ty.sty {
            ty::TyAdt(adt, _) => match adt.adt_kind() {
                AdtKind::Struct => Def::Struct,
                AdtKind::Enum => Def::Enum,
                AdtKind::Union => Def::Union,
            }
            _ => panic!("Unexpected type {:?}", def_id),
        };

        self.get_auto_trait_impls(def_id, def_ctor, None)
    }

    pub fn get_with_node_id(&self, id: ast::NodeId, name: String) -> Vec<Item> {
        let item = &self.cx.tcx.hir.expect_item(id).node;
        let did = self.cx.tcx.hir.local_def_id(id);

        let def_ctor = match *item {
            hir::ItemStruct(_, _) => Def::Struct,
            hir::ItemUnion(_, _) => Def::Union,
            hir::ItemEnum(_, _) => Def::Enum,
            _ => panic!("Unexpected type {:?} {:?}", item, id),
        };

        self.get_auto_trait_impls(did, def_ctor, Some(name))
    }

    pub fn get_auto_trait_impls(
        &self,
        def_id: DefId,
        def_ctor: fn(DefId) -> Def,
        name: Option<String>,
    ) -> Vec<Item> {
        if self.cx
            .tcx
            .get_attrs(def_id)
            .lists("doc")
            .has_word("hidden")
        {
            debug!(
                "get_auto_trait_impls(def_id={:?}, def_ctor={:?}): item has doc('hidden'), \
                 aborting",
                def_id, def_ctor
            );
            return Vec::new();
        }

        let tcx = self.cx.tcx;
        let generics = self.cx.tcx.generics_of(def_id);

        debug!(
            "get_auto_trait_impls(def_id={:?}, def_ctor={:?}, generics={:?}",
            def_id, def_ctor, generics
        );
        let auto_traits: Vec<_> = self.cx
            .send_trait
            .and_then(|send_trait| {
                self.get_auto_trait_impl_for(
                    def_id,
                    name.clone(),
                    generics.clone(),
                    def_ctor,
                    send_trait,
                )
            })
            .into_iter()
            .chain(self.get_auto_trait_impl_for(
                def_id,
                name.clone(),
                generics.clone(),
                def_ctor,
                tcx.require_lang_item(lang_items::SyncTraitLangItem),
            ).into_iter())
            .collect();

        debug!(
            "get_auto_traits: type {:?} auto_traits {:?}",
            def_id, auto_traits
        );
        auto_traits
    }

    fn get_auto_trait_impl_for(
        &self,
        def_id: DefId,
        name: Option<String>,
        generics: ty::Generics,
        def_ctor: fn(DefId) -> Def,
        trait_def_id: DefId,
    ) -> Option<Item> {
        if !self.cx
            .generated_synthetics
            .borrow_mut()
            .insert((def_id, trait_def_id))
        {
            debug!(
                "get_auto_trait_impl_for(def_id={:?}, generics={:?}, def_ctor={:?}, \
                 trait_def_id={:?}): already generated, aborting",
                def_id, generics, def_ctor, trait_def_id
            );
            return None;
        }

        let result = self.find_auto_trait_generics(def_id, trait_def_id, &generics);

        if result.is_auto() {
            let trait_ = hir::TraitRef {
                path: get_path_for_type(self.cx.tcx, trait_def_id, hir::def::Def::Trait),
                ref_id: ast::DUMMY_NODE_ID,
            };

            let polarity;

            let new_generics = match result {
                AutoTraitResult::PositiveImpl(new_generics) => {
                    polarity = None;
                    new_generics
                }
                AutoTraitResult::NegativeImpl => {
                    polarity = Some(ImplPolarity::Negative);

                    // For negative impls, we use the generic params, but *not* the predicates,
                    // from the original type. Otherwise, the displayed impl appears to be a
                    // conditional negative impl, when it's really unconditional.
                    //
                    // For example, consider the struct Foo<T: Copy>(*mut T). Using
                    // the original predicates in our impl would cause us to generate
                    // `impl !Send for Foo<T: Copy>`, which makes it appear that Foo
                    // implements Send where T is not copy.
                    //
                    // Instead, we generate `impl !Send for Foo<T>`, which better
                    // expresses the fact that `Foo<T>` never implements `Send`,
                    // regardless of the choice of `T`.
                    let real_generics = (&generics, &Default::default());

                    // Clean the generics, but ignore the '?Sized' bounds generated
                    // by the `Clean` impl
                    let clean_generics = real_generics.clean(self.cx);

                    Generics {
                        params: clean_generics.params,
                        where_predicates: Vec::new(),
                    }
                }
                _ => unreachable!(),
            };

            let path = get_path_for_type(self.cx.tcx, def_id, def_ctor);
            let mut segments = path.segments.into_vec();
            let last = segments.pop().unwrap();

            let real_name = name.as_ref().map(|n| Symbol::from(n.as_str()));

            segments.push(hir::PathSegment::new(
                real_name.unwrap_or(last.name),
                self.generics_to_path_params(generics.clone()),
                false,
            ));

            let new_path = hir::Path {
                span: path.span,
                def: path.def,
                segments: HirVec::from_vec(segments),
            };

            let ty = hir::Ty {
                id: ast::DUMMY_NODE_ID,
                node: hir::Ty_::TyPath(hir::QPath::Resolved(None, P(new_path))),
                span: DUMMY_SP,
                hir_id: hir::DUMMY_HIR_ID,
            };

            return Some(Item {
                source: Span::empty(),
                name: None,
                attrs: Default::default(),
                visibility: None,
                def_id: self.next_def_id(def_id.krate),
                stability: None,
                deprecation: None,
                inner: ImplItem(Impl {
                    unsafety: hir::Unsafety::Normal,
                    generics: new_generics,
                    provided_trait_methods: FxHashSet(),
                    trait_: Some(trait_.clean(self.cx)),
                    for_: ty.clean(self.cx),
                    items: Vec::new(),
                    polarity,
                    synthetic: true,
                }),
            });
        }
        None
    }

    fn generics_to_path_params(&self, generics: ty::Generics) -> hir::PathParameters {
        let lifetimes = HirVec::from_vec(
            generics
                .regions
                .iter()
                .map(|p| {
                    let name = if p.name == "" {
                        hir::LifetimeName::Static
                    } else {
                        hir::LifetimeName::Name(p.name)
                    };

                    hir::Lifetime {
                        id: ast::DUMMY_NODE_ID,
                        span: DUMMY_SP,
                        name,
                    }
                })
                .collect(),
        );
        let types = HirVec::from_vec(
            generics
                .types
                .iter()
                .map(|p| P(self.ty_param_to_ty(p.clone())))
                .collect(),
        );

        hir::PathParameters {
            lifetimes: lifetimes,
            types: types,
            bindings: HirVec::new(),
            parenthesized: false,
        }
    }

    fn ty_param_to_ty(&self, param: ty::TypeParameterDef) -> hir::Ty {
        debug!("ty_param_to_ty({:?}) {:?}", param, param.def_id);
        hir::Ty {
            id: ast::DUMMY_NODE_ID,
            node: hir::Ty_::TyPath(hir::QPath::Resolved(
                None,
                P(hir::Path {
                    span: DUMMY_SP,
                    def: Def::TyParam(param.def_id),
                    segments: HirVec::from_vec(vec![hir::PathSegment::from_name(param.name)]),
                }),
            )),
            span: DUMMY_SP,
            hir_id: hir::DUMMY_HIR_ID,
        }
    }

    fn find_auto_trait_generics(
        &self,
        did: DefId,
        trait_did: DefId,
        generics: &ty::Generics,
    ) -> AutoTraitResult {
        let tcx = self.cx.tcx;
        let ty = self.cx.tcx.type_of(did);

        let orig_params = tcx.param_env(did);

        let trait_ref = ty::TraitRef {
            def_id: trait_did,
            substs: tcx.mk_substs_trait(ty, &[]),
        };

        let trait_pred = ty::Binder(trait_ref);

        let bail_out = tcx.infer_ctxt().enter(|infcx| {
            let mut selcx = SelectionContext::with_negative(&infcx, true);
            let result = selcx.select(&Obligation::new(
                ObligationCause::dummy(),
                orig_params,
                trait_pred.to_poly_trait_predicate(),
            ));
            match result {
                Ok(Some(Vtable::VtableImpl(_))) => {
                    debug!(
                        "find_auto_trait_generics(did={:?}, trait_did={:?}, generics={:?}): \
                         manual impl found, bailing out",
                        did, trait_did, generics
                    );
                    return true;
                }
                _ => return false,
            };
        });

        // If an explicit impl exists, it always takes priority over an auto impl
        if bail_out {
            return AutoTraitResult::ExplicitImpl;
        }

        return tcx.infer_ctxt().enter(|mut infcx| {
            let mut fresh_preds = FxHashSet();

            // Due to the way projections are handled by SelectionContext, we need to run
            // evaluate_predicates twice: once on the original param env, and once on the result of
            // the first evaluate_predicates call.
            //
            // The problem is this: most of rustc, including SelectionContext and traits::project,
            // are designed to work with a concrete usage of a type (e.g. Vec<u8>
            // fn<T>() { Vec<T> }. This information will generally never change - given
            // the 'T' in fn<T>() { ... }, we'll never know anything else about 'T'.
            // If we're unable to prove that 'T' implements a particular trait, we're done -
            // there's nothing left to do but error out.
            //
            // However, synthesizing an auto trait impl works differently. Here, we start out with
            // a set of initial conditions - the ParamEnv of the struct/enum/union we're dealing
            // with - and progressively discover the conditions we need to fulfill for it to
            // implement a certain auto trait. This ends up breaking two assumptions made by trait
            // selection and projection:
            //
            // * We can always cache the result of a particular trait selection for the lifetime of
            // an InfCtxt
            // * Given a projection bound such as '<T as SomeTrait>::SomeItem = K', if 'T:
            // SomeTrait' doesn't hold, then we don't need to care about the 'SomeItem = K'
            //
            // We fix the first assumption by manually clearing out all of the InferCtxt's caches
            // in between calls to SelectionContext.select. This allows us to keep all of the
            // intermediate types we create bound to the 'tcx lifetime, rather than needing to lift
            // them between calls.
            //
            // We fix the second assumption by reprocessing the result of our first call to
            // evaluate_predicates. Using the example of '<T as SomeTrait>::SomeItem = K', our first
            // pass will pick up 'T: SomeTrait', but not 'SomeItem = K'. On our second pass,
            // traits::project will see that 'T: SomeTrait' is in our ParamEnv, allowing
            // SelectionContext to return it back to us.

            let (new_env, user_env) = match self.evaluate_predicates(
                &mut infcx,
                did,
                trait_did,
                ty,
                orig_params.clone(),
                orig_params,
                &mut fresh_preds,
                false,
            ) {
                Some(e) => e,
                None => return AutoTraitResult::NegativeImpl,
            };

            let (full_env, full_user_env) = self.evaluate_predicates(
                &mut infcx,
                did,
                trait_did,
                ty,
                new_env.clone(),
                user_env,
                &mut fresh_preds,
                true,
            ).unwrap_or_else(|| {
                panic!(
                    "Failed to fully process: {:?} {:?} {:?}",
                    ty, trait_did, orig_params
                )
            });

            debug!(
                "find_auto_trait_generics(did={:?}, trait_did={:?}, generics={:?}): fulfilling \
                 with {:?}",
                did, trait_did, generics, full_env
            );
            infcx.clear_caches();

            // At this point, we already have all of the bounds we need. FulfillmentContext is used
            // to store all of the necessary region/lifetime bounds in the InferContext, as well as
            // an additional sanity check.
            let mut fulfill = FulfillmentContext::new();
            fulfill.register_bound(
                &infcx,
                full_env,
                ty,
                trait_did,
                ObligationCause::misc(DUMMY_SP, ast::DUMMY_NODE_ID),
            );
            fulfill.select_all_or_error(&infcx).unwrap_or_else(|e| {
                panic!(
                    "Unable to fulfill trait {:?} for '{:?}': {:?}",
                    trait_did, ty, e
                )
            });

            let names_map: FxHashMap<String, Lifetime> = generics
                .regions
                .iter()
                .map(|l| (l.name.as_str().to_string(), l.clean(self.cx)))
                .collect();

            let body_ids: FxHashSet<_> = infcx
                .region_obligations
                .borrow()
                .iter()
                .map(|&(id, _)| id)
                .collect();

            for id in body_ids {
                infcx.process_registered_region_obligations(&[], None, full_env.clone(), id);
            }

            let region_data = infcx
                .borrow_region_constraints()
                .region_constraint_data()
                .clone();

            let lifetime_predicates = self.handle_lifetimes(&region_data, &names_map);
            let vid_to_region = self.map_vid_to_region(&region_data);

            debug!(
                "find_auto_trait_generics(did={:?}, trait_did={:?}, generics={:?}): computed \
                 lifetime information '{:?}' '{:?}'",
                did, trait_did, generics, lifetime_predicates, vid_to_region
            );

            let new_generics = self.param_env_to_generics(
                infcx.tcx,
                did,
                full_user_env,
                generics.clone(),
                lifetime_predicates,
                vid_to_region,
            );
            debug!(
                "find_auto_trait_generics(did={:?}, trait_did={:?}, generics={:?}): finished with \
                 {:?}",
                did, trait_did, generics, new_generics
            );
            return AutoTraitResult::PositiveImpl(new_generics);
        });
    }

    fn clean_pred<'c, 'd, 'cx>(
        &self,
        infcx: &InferCtxt<'c, 'd, 'cx>,
        p: ty::Predicate<'cx>,
    ) -> ty::Predicate<'cx> {
        infcx.freshen(p)
    }

    fn evaluate_nested_obligations<'b, 'c, 'd, 'cx,
                                    T: Iterator<Item = Obligation<'cx, ty::Predicate<'cx>>>>(
        &self,
        ty: ty::Ty,
        nested: T,
        computed_preds: &'b mut FxHashSet<ty::Predicate<'cx>>,
        fresh_preds: &'b mut FxHashSet<ty::Predicate<'cx>>,
        predicates: &'b mut VecDeque<ty::PolyTraitPredicate<'cx>>,
        select: &mut traits::SelectionContext<'c, 'd, 'cx>,
        only_projections: bool,
    ) -> bool {
        let dummy_cause = ObligationCause::misc(DUMMY_SP, ast::DUMMY_NODE_ID);

        for (obligation, predicate) in nested
            .filter(|o| o.recursion_depth == 1)
            .map(|o| (o.clone(), o.predicate.clone()))
        {
            let is_new_pred =
                fresh_preds.insert(self.clean_pred(select.infcx(), predicate.clone()));

            match &predicate {
                &ty::Predicate::Trait(ref p) => {
                    let substs = &p.skip_binder().trait_ref.substs;

                    if self.is_of_param(substs) && !only_projections && is_new_pred {
                        computed_preds.insert(predicate);
                    }
                    predicates.push_back(p.clone());
                }
                &ty::Predicate::Projection(p) => {
                    // If the projection isn't all type vars, then
                    // we don't want to add it as a bound
                    if self.is_of_param(p.skip_binder().projection_ty.substs) && is_new_pred {
                        computed_preds.insert(predicate);
                    } else {
                        match traits::poly_project_and_unify_type(
                            select,
                            &obligation.with(p.clone()),
                        ) {
                            Err(e) => {
                                debug!(
                                    "evaluate_nested_obligations: Unable to unify predicate \
                                     '{:?}' '{:?}', bailing out",
                                    ty, e
                                );
                                return false;
                            }
                            Ok(Some(v)) => {
                                if !self.evaluate_nested_obligations(
                                    ty,
                                    v.clone().iter().cloned(),
                                    computed_preds,
                                    fresh_preds,
                                    predicates,
                                    select,
                                    only_projections,
                                ) {
                                    return false;
                                }
                            }
                            Ok(None) => {
                                panic!("Unexpected result when selecting {:?} {:?}", ty, obligation)
                            }
                        }
                    }
                }
                &ty::Predicate::RegionOutlives(ref binder) => {
                    if let Err(_) = select
                        .infcx()
                        .region_outlives_predicate(&dummy_cause, binder)
                    {
                        return false;
                    }
                }
                &ty::Predicate::TypeOutlives(ref binder) => {
                    match (
                        binder.no_late_bound_regions(),
                        binder.map_bound_ref(|pred| pred.0).no_late_bound_regions(),
                    ) {
                        (None, Some(t_a)) => {
                            select.infcx().register_region_obligation(
                                ast::DUMMY_NODE_ID,
                                RegionObligation {
                                    sup_type: t_a,
                                    sub_region: select.infcx().tcx.types.re_static,
                                    cause: dummy_cause.clone(),
                                },
                            );
                        }
                        (Some(ty::OutlivesPredicate(t_a, r_b)), _) => {
                            select.infcx().register_region_obligation(
                                ast::DUMMY_NODE_ID,
                                RegionObligation {
                                    sup_type: t_a,
                                    sub_region: r_b,
                                    cause: dummy_cause.clone(),
                                },
                            );
                        }
                        _ => {}
                    };
                }
                _ => panic!("Unexpected predicate {:?} {:?}", ty, predicate),
            };
        }
        return true;
    }

    // The core logic responsible for computing the bounds for our synthesized impl.
    //
    // To calculate the bounds, we call SelectionContext.select in a loop. Like FulfillmentContext,
    // we recursively select the nested obligations of predicates we encounter. However, whenver we
    // encounter an UnimplementedError involving a type parameter, we add it to our ParamEnv. Since
    // our goal is to determine when a particular type implements an auto trait, Unimplemented
    // errors tell us what conditions need to be met.
    //
    // This method ends up working somewhat similary to FulfillmentContext, but with a few key
    // differences. FulfillmentContext works under the assumption that it's dealing with concrete
    // user code. According, it considers all possible ways that a Predicate could be met - which
    // isn't always what we want for a synthesized impl. For example, given the predicate 'T:
    // Iterator', FulfillmentContext can end up reporting an Unimplemented error for T:
    // IntoIterator - since there's an implementation of Iteratpr where T: IntoIterator,
    // FulfillmentContext will drive SelectionContext to consider that impl before giving up. If we
    // were to rely on FulfillmentContext's decision, we might end up synthesizing an impl like
    // this:
    // 'impl<T> Send for Foo<T> where T: IntoIterator'
    //
    // While it might be technically true that Foo implements Send where T: IntoIterator,
    // the bound is overly restrictive - it's really only necessary that T: Iterator.
    //
    // For this reason, evaluate_predicates handles predicates with type variables specially. When
    // we encounter an Unimplemented error for a bound such as 'T: Iterator', we immediately add it
    // to our ParamEnv, and add it to our stack for recursive evaluation. When we later select it,
    // we'll pick up any nested bounds, without ever inferring that 'T: IntoIterator' needs to
    // hold.
    //
    // One additonal consideration is supertrait bounds. Normally, a ParamEnv is only ever
    // consutrcted once for a given type. As part of the construction process, the ParamEnv will
    // have any supertrait bounds normalized - e.g. if we have a type 'struct Foo<T: Copy>', the
    // ParamEnv will contain 'T: Copy' and 'T: Clone', since 'Copy: Clone'. When we construct our
    // own ParamEnv, we need to do this outselves, through traits::elaborate_predicates, or else
    // SelectionContext will choke on the missing predicates. However, this should never show up in
    // the final synthesized generics: we don't want our generated docs page to contain something
    // like 'T: Copy + Clone', as that's redundant. Therefore, we keep track of a separate
    // 'user_env', which only holds the predicates that will actually be displayed to the user.
    fn evaluate_predicates<'b, 'gcx, 'c>(
        &self,
        infcx: &mut InferCtxt<'b, 'tcx, 'c>,
        ty_did: DefId,
        trait_did: DefId,
        ty: ty::Ty<'c>,
        param_env: ty::ParamEnv<'c>,
        user_env: ty::ParamEnv<'c>,
        fresh_preds: &mut FxHashSet<ty::Predicate<'c>>,
        only_projections: bool,
    ) -> Option<(ty::ParamEnv<'c>, ty::ParamEnv<'c>)> {
        let tcx = infcx.tcx;

        let mut select = traits::SelectionContext::new(&infcx);

        let mut already_visited = FxHashSet();
        let mut predicates = VecDeque::new();
        predicates.push_back(ty::Binder(ty::TraitPredicate {
            trait_ref: ty::TraitRef {
                def_id: trait_did,
                substs: infcx.tcx.mk_substs_trait(ty, &[]),
            },
        }));

        let mut computed_preds: FxHashSet<_> = param_env.caller_bounds.iter().cloned().collect();
        let mut user_computed_preds: FxHashSet<_> =
            user_env.caller_bounds.iter().cloned().collect();

        let mut new_env = param_env.clone();
        let dummy_cause = ObligationCause::misc(DUMMY_SP, ast::DUMMY_NODE_ID);

        while let Some(pred) = predicates.pop_front() {
            infcx.clear_caches();

            if !already_visited.insert(pred.clone()) {
                continue;
            }

            let result = select.select(&Obligation::new(dummy_cause.clone(), new_env, pred));

            match &result {
                &Ok(Some(ref vtable)) => {
                    let obligations = vtable.clone().nested_obligations().into_iter();

                    if !self.evaluate_nested_obligations(
                        ty,
                        obligations,
                        &mut user_computed_preds,
                        fresh_preds,
                        &mut predicates,
                        &mut select,
                        only_projections,
                    ) {
                        return None;
                    }
                }
                &Ok(None) => {}
                &Err(SelectionError::Unimplemented) => {
                    if self.is_of_param(pred.skip_binder().trait_ref.substs) {
                        already_visited.remove(&pred);
                        user_computed_preds.insert(ty::Predicate::Trait(pred.clone()));
                        predicates.push_back(pred);
                    } else {
                        debug!(
                            "evaluate_nested_obligations: Unimplemented found, bailing: {:?} {:?} \
                             {:?}",
                            ty,
                            pred,
                            pred.skip_binder().trait_ref.substs
                        );
                        return None;
                    }
                }
                _ => panic!("Unexpected error for '{:?}': {:?}", ty, result),
            };

            computed_preds.extend(user_computed_preds.iter().cloned());
            let normalized_preds =
                traits::elaborate_predicates(tcx, computed_preds.clone().into_iter().collect());
            new_env = ty::ParamEnv::new(
                tcx.mk_predicates(normalized_preds),
                param_env.reveal,
                ty::UniverseIndex::ROOT,
            );
        }

        let final_user_env = ty::ParamEnv::new(
            tcx.mk_predicates(user_computed_preds.into_iter()),
            user_env.reveal,
            ty::UniverseIndex::ROOT,
        );
        debug!(
            "evaluate_nested_obligations(ty_did={:?}, trait_did={:?}): succeeded with '{:?}' \
             '{:?}'",
            ty_did, trait_did, new_env, final_user_env
        );

        return Some((new_env, final_user_env));
    }

    fn is_of_param(&self, substs: &Substs) -> bool {
        if substs.is_noop() {
            return false;
        }

        return match substs.type_at(0).sty {
            ty::TyParam(_) => true,
            ty::TyProjection(p) => self.is_of_param(p.substs),
            _ => false,
        };
    }

    fn get_lifetime(&self, region: Region, names_map: &FxHashMap<String, Lifetime>) -> Lifetime {
        self.region_name(region)
            .map(|name| {
                names_map.get(&name).unwrap_or_else(|| {
                    panic!("Missing lifetime with name {:?} for {:?}", name, region)
                })
            })
            .unwrap_or(&Lifetime::statik())
            .clone()
    }

    fn region_name(&self, region: Region) -> Option<String> {
        match region {
            &ty::ReEarlyBound(r) => Some(r.name.as_str().to_string()),
            _ => None,
        }
    }

    // This is very similar to handle_lifetimes. However, instead of matching ty::Region's
    // to each other, we match ty::RegionVid's to ty::Region's
    fn map_vid_to_region<'cx>(
        &self,
        regions: &RegionConstraintData<'cx>,
    ) -> FxHashMap<ty::RegionVid, ty::Region<'cx>> {
        let mut vid_map: FxHashMap<RegionTarget<'cx>, RegionDeps<'cx>> = FxHashMap();
        let mut finished_map = FxHashMap();

        for constraint in regions.constraints.keys() {
            match constraint {
                &Constraint::VarSubVar(r1, r2) => {
                    {
                        let deps1 = vid_map
                            .entry(RegionTarget::RegionVid(r1))
                            .or_insert_with(|| Default::default());
                        deps1.larger.insert(RegionTarget::RegionVid(r2));
                    }

                    let deps2 = vid_map
                        .entry(RegionTarget::RegionVid(r2))
                        .or_insert_with(|| Default::default());
                    deps2.smaller.insert(RegionTarget::RegionVid(r1));
                }
                &Constraint::RegSubVar(region, vid) => {
                    {
                        let deps1 = vid_map
                            .entry(RegionTarget::Region(region))
                            .or_insert_with(|| Default::default());
                        deps1.larger.insert(RegionTarget::RegionVid(vid));
                    }

                    let deps2 = vid_map
                        .entry(RegionTarget::RegionVid(vid))
                        .or_insert_with(|| Default::default());
                    deps2.smaller.insert(RegionTarget::Region(region));
                }
                &Constraint::VarSubReg(vid, region) => {
                    finished_map.insert(vid, region);
                }
                &Constraint::RegSubReg(r1, r2) => {
                    {
                        let deps1 = vid_map
                            .entry(RegionTarget::Region(r1))
                            .or_insert_with(|| Default::default());
                        deps1.larger.insert(RegionTarget::Region(r2));
                    }

                    let deps2 = vid_map
                        .entry(RegionTarget::Region(r2))
                        .or_insert_with(|| Default::default());
                    deps2.smaller.insert(RegionTarget::Region(r1));
                }
            }
        }

        while !vid_map.is_empty() {
            let target = vid_map.keys().next().expect("Keys somehow empty").clone();
            let deps = vid_map.remove(&target).expect("Entry somehow missing");

            for smaller in deps.smaller.iter() {
                for larger in deps.larger.iter() {
                    match (smaller, larger) {
                        (&RegionTarget::Region(_), &RegionTarget::Region(_)) => {
                            if let Entry::Occupied(v) = vid_map.entry(*smaller) {
                                let smaller_deps = v.into_mut();
                                smaller_deps.larger.insert(*larger);
                                smaller_deps.larger.remove(&target);
                            }

                            if let Entry::Occupied(v) = vid_map.entry(*larger) {
                                let larger_deps = v.into_mut();
                                larger_deps.smaller.insert(*smaller);
                                larger_deps.smaller.remove(&target);
                            }
                        }
                        (&RegionTarget::RegionVid(v1), &RegionTarget::Region(r1)) => {
                            finished_map.insert(v1, r1);
                        }
                        (&RegionTarget::Region(_), &RegionTarget::RegionVid(_)) => {
                            // Do nothing - we don't care about regions that are smaller than vids
                        }
                        (&RegionTarget::RegionVid(_), &RegionTarget::RegionVid(_)) => {
                            if let Entry::Occupied(v) = vid_map.entry(*smaller) {
                                let smaller_deps = v.into_mut();
                                smaller_deps.larger.insert(*larger);
                                smaller_deps.larger.remove(&target);
                            }

                            if let Entry::Occupied(v) = vid_map.entry(*larger) {
                                let larger_deps = v.into_mut();
                                larger_deps.smaller.insert(*smaller);
                                larger_deps.smaller.remove(&target);
                            }
                        }
                    }
                }
            }
        }
        finished_map
    }

    // This method calculates two things: Lifetime constraints of the form 'a: 'b,
    // and region constraints of the form ReVar: 'a
    //
    // This is essentially a simplified version of lexical_region_resolve. However,
    // handle_lifetimes determines what *needs be* true in order for an impl to hold.
    // lexical_region_resolve, along with much of the rest of the compiler, is concerned
    // with determining if a given set up constraints/predicates *are* met, given some
    // starting conditions (e.g. user-provided code). For this reason, it's easier
    // to perform the calculations we need on our own, rather than trying to make
    // existing inference/solver code do what we want.
    fn handle_lifetimes<'cx>(
        &self,
        regions: &RegionConstraintData<'cx>,
        names_map: &FxHashMap<String, Lifetime>,
    ) -> Vec<WherePredicate> {
        // Our goal is to 'flatten' the list of constraints by eliminating
        // all intermediate RegionVids. At the end, all constraints should
        // be between Regions (aka region variables). This gives us the information
        // we need to create the Generics.
        let mut finished = FxHashMap();

        let mut vid_map: FxHashMap<RegionTarget, RegionDeps> = FxHashMap();

        // Flattening is done in two parts. First, we insert all of the constraints
        // into a map. Each RegionTarget (either a RegionVid or a Region) maps
        // to its smaller and larger regions. Note that 'larger' regions correspond
        // to sub-regions in Rust code (e.g. in 'a: 'b, 'a is the larger region).
        for constraint in regions.constraints.keys() {
            match constraint {
                &Constraint::VarSubVar(r1, r2) => {
                    {
                        let deps1 = vid_map
                            .entry(RegionTarget::RegionVid(r1))
                            .or_insert_with(|| Default::default());
                        deps1.larger.insert(RegionTarget::RegionVid(r2));
                    }

                    let deps2 = vid_map
                        .entry(RegionTarget::RegionVid(r2))
                        .or_insert_with(|| Default::default());
                    deps2.smaller.insert(RegionTarget::RegionVid(r1));
                }
                &Constraint::RegSubVar(region, vid) => {
                    let deps = vid_map
                        .entry(RegionTarget::RegionVid(vid))
                        .or_insert_with(|| Default::default());
                    deps.smaller.insert(RegionTarget::Region(region));
                }
                &Constraint::VarSubReg(vid, region) => {
                    let deps = vid_map
                        .entry(RegionTarget::RegionVid(vid))
                        .or_insert_with(|| Default::default());
                    deps.larger.insert(RegionTarget::Region(region));
                }
                &Constraint::RegSubReg(r1, r2) => {
                    // The constraint is already in the form that we want, so we're done with it
                    // Desired order is 'larger, smaller', so flip then
                    if self.region_name(r1) != self.region_name(r2) {
                        finished
                            .entry(self.region_name(r2).unwrap())
                            .or_insert_with(|| Vec::new())
                            .push(r1);
                    }
                }
            }
        }

        // Here, we 'flatten' the map one element at a time.
        // All of the element's sub and super regions are connected
        // to each other. For example, if we have a graph that looks like this:
        //
        // (A, B) - C - (D, E)
        // Where (A, B) are subregions, and (D,E) are super-regions
        //
        // then after deleting 'C', the graph will look like this:
        //  ... - A - (D, E ...)
        //  ... - B - (D, E, ...)
        //  (A, B, ...) - D - ...
        //  (A, B, ...) - E - ...
        //
        //  where '...' signifies the existing sub and super regions of an entry
        //  When two adjacent ty::Regions are encountered, we've computed a final
        //  constraint, and add it to our list. Since we make sure to never re-add
        //  deleted items, this process will always finish.
        while !vid_map.is_empty() {
            let target = vid_map.keys().next().expect("Keys somehow empty").clone();
            let deps = vid_map.remove(&target).expect("Entry somehow missing");

            for smaller in deps.smaller.iter() {
                for larger in deps.larger.iter() {
                    match (smaller, larger) {
                        (&RegionTarget::Region(r1), &RegionTarget::Region(r2)) => {
                            if self.region_name(r1) != self.region_name(r2) {
                                finished
                                    .entry(self.region_name(r2).unwrap())
                                    .or_insert_with(|| Vec::new())
                                    .push(r1) // Larger, smaller
                            }
                        }
                        (&RegionTarget::RegionVid(_), &RegionTarget::Region(_)) => {
                            if let Entry::Occupied(v) = vid_map.entry(*smaller) {
                                let smaller_deps = v.into_mut();
                                smaller_deps.larger.insert(*larger);
                                smaller_deps.larger.remove(&target);
                            }
                        }
                        (&RegionTarget::Region(_), &RegionTarget::RegionVid(_)) => {
                            if let Entry::Occupied(v) = vid_map.entry(*larger) {
                                let deps = v.into_mut();
                                deps.smaller.insert(*smaller);
                                deps.smaller.remove(&target);
                            }
                        }
                        (&RegionTarget::RegionVid(_), &RegionTarget::RegionVid(_)) => {
                            if let Entry::Occupied(v) = vid_map.entry(*smaller) {
                                let smaller_deps = v.into_mut();
                                smaller_deps.larger.insert(*larger);
                                smaller_deps.larger.remove(&target);
                            }

                            if let Entry::Occupied(v) = vid_map.entry(*larger) {
                                let larger_deps = v.into_mut();
                                larger_deps.smaller.insert(*smaller);
                                larger_deps.smaller.remove(&target);
                            }
                        }
                    }
                }
            }
        }

        let lifetime_predicates = names_map
            .iter()
            .flat_map(|(name, lifetime)| {
                let empty = Vec::new();
                let bounds: FxHashSet<Lifetime> = finished
                    .get(name)
                    .unwrap_or(&empty)
                    .iter()
                    .map(|region| self.get_lifetime(region, names_map))
                    .collect();

                if bounds.is_empty() {
                    return None;
                }
                Some(WherePredicate::RegionPredicate {
                    lifetime: lifetime.clone(),
                    bounds: bounds.into_iter().collect(),
                })
            })
            .collect();

        lifetime_predicates
    }

    fn extract_for_generics<'b, 'c, 'd>(
        &self,
        tcx: TyCtxt<'b, 'c, 'd>,
        pred: ty::Predicate<'d>,
    ) -> FxHashSet<GenericParam> {
        pred.walk_tys()
            .flat_map(|t| {
                let mut regions = FxHashSet();
                tcx.collect_regions(&t, &mut regions);

                regions.into_iter().flat_map(|r| {
                    match r {
                        // We only care about late bound regions, as we need to add them
                        // to the 'for<>' section
                        &ty::ReLateBound(_, ty::BoundRegion::BrNamed(_, name)) => {
                            Some(GenericParam::Lifetime(Lifetime(name.as_str().to_string())))
                        }
                        &ty::ReVar(_) | &ty::ReEarlyBound(_) => None,
                        _ => panic!("Unexpected region type {:?}", r),
                    }
                })
            })
            .collect()
    }

    fn make_final_bounds<'b, 'c, 'cx>(
        &self,
        ty_to_bounds: FxHashMap<Type, FxHashSet<TyParamBound>>,
        ty_to_fn: FxHashMap<Type, (Option<PolyTrait>, Option<Type>)>,
        lifetime_to_bounds: FxHashMap<Lifetime, FxHashSet<Lifetime>>,
    ) -> Vec<WherePredicate> {
        ty_to_bounds
            .into_iter()
            .flat_map(|(ty, mut bounds)| {
                if let Some(data) = ty_to_fn.get(&ty) {
                    let (poly_trait, output) =
                        (data.0.as_ref().unwrap().clone(), data.1.as_ref().cloned());
                    let new_ty = match &poly_trait.trait_ {
                        &Type::ResolvedPath {
                            ref path,
                            ref typarams,
                            ref did,
                            ref is_generic,
                        } => {
                            let mut new_path = path.clone();
                            let last_segment = new_path.segments.pop().unwrap();

                            let (old_input, old_output) = match last_segment.params {
                                PathParameters::AngleBracketed { types, .. } => (types, None),
                                PathParameters::Parenthesized { inputs, output, .. } => {
                                    (inputs, output)
                                }
                            };

                            if old_output.is_some() && old_output != output {
                                panic!(
                                    "Output mismatch for {:?} {:?} {:?}",
                                    ty, old_output, data.1
                                );
                            }

                            let new_params = PathParameters::Parenthesized {
                                inputs: old_input,
                                output,
                            };

                            new_path.segments.push(PathSegment {
                                name: last_segment.name,
                                params: new_params,
                            });

                            Type::ResolvedPath {
                                path: new_path,
                                typarams: typarams.clone(),
                                did: did.clone(),
                                is_generic: *is_generic,
                            }
                        }
                        _ => panic!("Unexpected data: {:?}, {:?}", ty, data),
                    };
                    bounds.insert(TyParamBound::TraitBound(
                        PolyTrait {
                            trait_: new_ty,
                            generic_params: poly_trait.generic_params,
                        },
                        hir::TraitBoundModifier::None,
                    ));
                }
                if bounds.is_empty() {
                    return None;
                }

                let mut bounds_vec = bounds.into_iter().collect();
                self.sort_where_bounds(&mut bounds_vec);

                Some(WherePredicate::BoundPredicate {
                    ty,
                    bounds: bounds_vec,
                })
            })
            .chain(
                lifetime_to_bounds
                    .into_iter()
                    .filter(|&(_, ref bounds)| !bounds.is_empty())
                    .map(|(lifetime, bounds)| {
                        let mut bounds_vec = bounds.into_iter().collect();
                        self.sort_where_lifetimes(&mut bounds_vec);
                        WherePredicate::RegionPredicate {
                            lifetime,
                            bounds: bounds_vec,
                        }
                    }),
            )
            .collect()
    }

    // Converts the calculated ParamEnv and lifetime information to a clean::Generics, suitable for
    // display on the docs page. Cleaning the Predicates produces sub-optimal WherePredicate's,
    // so we fix them up:
    //
    // * Multiple bounds for the same type are coalesced into one: e.g. 'T: Copy', 'T: Debug'
    // becomes 'T: Copy + Debug'
    // * Fn bounds are handled specially - instead of leaving it as 'T: Fn(), <T as Fn::Output> =
    // K', we use the dedicated syntax 'T: Fn() -> K'
    // * We explcitly add a '?Sized' bound if we didn't find any 'Sized' predicates for a type
    fn param_env_to_generics<'b, 'c, 'cx>(
        &self,
        tcx: TyCtxt<'b, 'c, 'cx>,
        did: DefId,
        param_env: ty::ParamEnv<'cx>,
        type_generics: ty::Generics,
        mut existing_predicates: Vec<WherePredicate>,
        vid_to_region: FxHashMap<ty::RegionVid, ty::Region<'cx>>,
    ) -> Generics {
        debug!(
            "param_env_to_generics(did={:?}, param_env={:?}, type_generics={:?}, \
             existing_predicates={:?})",
            did, param_env, type_generics, existing_predicates
        );

        // The `Sized` trait must be handled specially, since we only only display it when
        // it is *not* required (i.e. '?Sized')
        let sized_trait = self.cx
            .tcx
            .require_lang_item(lang_items::SizedTraitLangItem);

        let mut replacer = RegionReplacer {
            vid_to_region: &vid_to_region,
            tcx,
        };

        let orig_bounds: FxHashSet<_> = self.cx.tcx.param_env(did).caller_bounds.iter().collect();
        let clean_where_predicates = param_env
            .caller_bounds
            .iter()
            .filter(|p| {
                !orig_bounds.contains(p) || match p {
                    &&ty::Predicate::Trait(pred) => pred.def_id() == sized_trait,
                    _ => false,
                }
            })
            .map(|p| {
                let replaced = p.fold_with(&mut replacer);
                (replaced.clone(), replaced.clean(self.cx))
            });

        let full_generics = (&type_generics, &tcx.predicates_of(did));
        let Generics {
            params: mut generic_params,
            ..
        } = full_generics.clean(self.cx);

        let mut has_sized = FxHashSet();
        let mut ty_to_bounds = FxHashMap();
        let mut lifetime_to_bounds = FxHashMap();
        let mut ty_to_traits: FxHashMap<Type, FxHashSet<Type>> = FxHashMap();

        let mut ty_to_fn: FxHashMap<Type, (Option<PolyTrait>, Option<Type>)> = FxHashMap();

        for (orig_p, p) in clean_where_predicates {
            match p {
                WherePredicate::BoundPredicate { ty, mut bounds } => {
                    // Writing a projection trait bound of the form
                    // <T as Trait>::Name : ?Sized
                    // is illegal, because ?Sized bounds can only
                    // be written in the (here, nonexistant) definition
                    // of the type.
                    // Therefore, we make sure that we never add a ?Sized
                    // bound for projections
                    match &ty {
                        &Type::QPath { .. } => {
                            has_sized.insert(ty.clone());
                        }
                        _ => {}
                    }

                    if bounds.is_empty() {
                        continue;
                    }

                    let mut for_generics = self.extract_for_generics(tcx, orig_p.clone());

                    assert!(bounds.len() == 1);
                    let mut b = bounds.pop().unwrap();

                    if b.is_sized_bound(self.cx) {
                        has_sized.insert(ty.clone());
                    } else if !b.get_trait_type()
                        .and_then(|t| {
                            ty_to_traits
                                .get(&ty)
                                .map(|bounds| bounds.contains(&strip_type(t.clone())))
                        })
                        .unwrap_or(false)
                    {
                        // If we've already added a projection bound for the same type, don't add
                        // this, as it would be a duplicate

                        // Handle any 'Fn/FnOnce/FnMut' bounds specially,
                        // as we want to combine them with any 'Output' qpaths
                        // later

                        let is_fn = match &mut b {
                            &mut TyParamBound::TraitBound(ref mut p, _) => {
                                // Insert regions into the for_generics hash map first, to ensure
                                // that we don't end up with duplicate bounds (e.g. for<'b, 'b>)
                                for_generics.extend(p.generic_params.clone());
                                p.generic_params = for_generics.into_iter().collect();
                                self.is_fn_ty(&tcx, &p.trait_)
                            }
                            _ => false,
                        };

                        let poly_trait = b.get_poly_trait().unwrap();

                        if is_fn {
                            ty_to_fn
                                .entry(ty.clone())
                                .and_modify(|e| *e = (Some(poly_trait.clone()), e.1.clone()))
                                .or_insert(((Some(poly_trait.clone())), None));

                            ty_to_bounds
                                .entry(ty.clone())
                                .or_insert_with(|| FxHashSet());
                        } else {
                            ty_to_bounds
                                .entry(ty.clone())
                                .or_insert_with(|| FxHashSet())
                                .insert(b.clone());
                        }
                    }
                }
                WherePredicate::RegionPredicate { lifetime, bounds } => {
                    lifetime_to_bounds
                        .entry(lifetime)
                        .or_insert_with(|| FxHashSet())
                        .extend(bounds);
                }
                WherePredicate::EqPredicate { lhs, rhs } => {
                    match &lhs {
                        &Type::QPath {
                            name: ref left_name,
                            ref self_type,
                            ref trait_,
                        } => {
                            let ty = &*self_type;
                            match **trait_ {
                                Type::ResolvedPath {
                                    path: ref trait_path,
                                    ref typarams,
                                    ref did,
                                    ref is_generic,
                                } => {
                                    let mut new_trait_path = trait_path.clone();

                                    if self.is_fn_ty(&tcx, trait_) && left_name == FN_OUTPUT_NAME {
                                        ty_to_fn
                                            .entry(*ty.clone())
                                            .and_modify(|e| *e = (e.0.clone(), Some(rhs.clone())))
                                            .or_insert((None, Some(rhs)));
                                        continue;
                                    }

                                    // FIXME: Remove this scope when NLL lands
                                    {
                                        let params =
                                            &mut new_trait_path.segments.last_mut().unwrap().params;

                                        match params {
                                            // Convert somethiung like '<T as Iterator::Item> = u8'
                                            // to 'T: Iterator<Item=u8>'
                                            &mut PathParameters::AngleBracketed {
                                                ref mut bindings,
                                                ..
                                            } => {
                                                bindings.push(TypeBinding {
                                                    name: left_name.clone(),
                                                    ty: rhs,
                                                });
                                            }
                                            &mut PathParameters::Parenthesized { .. } => {
                                                existing_predicates.push(
                                                    WherePredicate::EqPredicate {
                                                        lhs: lhs.clone(),
                                                        rhs,
                                                    },
                                                );
                                                continue; // If something other than a Fn ends up
                                                          // with parenthesis, leave it alone
                                            }
                                        }
                                    }

                                    let bounds = ty_to_bounds
                                        .entry(*ty.clone())
                                        .or_insert_with(|| FxHashSet());

                                    bounds.insert(TyParamBound::TraitBound(
                                        PolyTrait {
                                            trait_: Type::ResolvedPath {
                                                path: new_trait_path,
                                                typarams: typarams.clone(),
                                                did: did.clone(),
                                                is_generic: *is_generic,
                                            },
                                            generic_params: Vec::new(),
                                        },
                                        hir::TraitBoundModifier::None,
                                    ));

                                    // Remove any existing 'plain' bound (e.g. 'T: Iterator`) so
                                    // that we don't see a
                                    // duplicate bound like `T: Iterator + Iterator<Item=u8>`
                                    // on the docs page.
                                    bounds.remove(&TyParamBound::TraitBound(
                                        PolyTrait {
                                            trait_: *trait_.clone(),
                                            generic_params: Vec::new(),
                                        },
                                        hir::TraitBoundModifier::None,
                                    ));
                                    // Avoid creating any new duplicate bounds later in the outer
                                    // loop
                                    ty_to_traits
                                        .entry(*ty.clone())
                                        .or_insert_with(|| FxHashSet())
                                        .insert(*trait_.clone());
                                }
                                _ => panic!("Unexpected trait {:?} for {:?}", trait_, did),
                            }
                        }
                        _ => panic!("Unexpected LHS {:?} for {:?}", lhs, did),
                    }
                }
            };
        }

        let final_bounds = self.make_final_bounds(ty_to_bounds, ty_to_fn, lifetime_to_bounds);

        existing_predicates.extend(final_bounds);

        for p in generic_params.iter_mut() {
            match p {
                &mut GenericParam::Type(ref mut ty) => {
                    // We never want something like 'impl<T=Foo>'
                    ty.default.take();

                    let generic_ty = Type::Generic(ty.name.clone());

                    if !has_sized.contains(&generic_ty) {
                        ty.bounds.insert(0, TyParamBound::maybe_sized(self.cx));
                    }
                }
                _ => {}
            }
        }

        self.sort_where_predicates(&mut existing_predicates);

        Generics {
            params: generic_params,
            where_predicates: existing_predicates,
        }
    }

    // Ensure that the predicates are in a consistent order. The precise
    // ordering doesn't actually matter, but it's important that
    // a given set of predicates always appears in the same order -
    // both for visual consistency between 'rustdoc' runs, and to
    // make writing tests much easier
    #[inline]
    fn sort_where_predicates(&self, mut predicates: &mut Vec<WherePredicate>) {
        // We should never have identical bounds - and if we do,
        // they're visually identical as well. Therefore, using
        // an unstable sort is fine.
        self.unstable_debug_sort(&mut predicates);
    }

    // Ensure that the bounds are in a consistent order. The precise
    // ordering doesn't actually matter, but it's important that
    // a given set of bounds always appears in the same order -
    // both for visual consistency between 'rustdoc' runs, and to
    // make writing tests much easier
    #[inline]
    fn sort_where_bounds(&self, mut bounds: &mut Vec<TyParamBound>) {
        // We should never have identical bounds - and if we do,
        // they're visually identical as well. Therefore, using
        // an unstable sort is fine.
        self.unstable_debug_sort(&mut bounds);
    }

    #[inline]
    fn sort_where_lifetimes(&self, mut bounds: &mut Vec<Lifetime>) {
        // We should never have identical bounds - and if we do,
        // they're visually identical as well. Therefore, using
        // an unstable sort is fine.
        self.unstable_debug_sort(&mut bounds);
    }

    // This might look horrendously hacky, but it's actually not that bad.
    //
    // For performance reasons, we use several different FxHashMaps
    // in the process of computing the final set of where predicates.
    // However, the iteration order of a HashMap is completely unspecified.
    // In fact, the iteration of an FxHashMap can even vary between platforms,
    // since FxHasher has different behavior for 32-bit and 64-bit platforms.
    //
    // Obviously, it's extremely undesireable for documentation rendering
    // to be depndent on the platform it's run on. Apart from being confusing
    // to end users, it makes writing tests much more difficult, as predicates
    // can appear in any order in the final result.
    //
    // To solve this problem, we sort WherePredicates and TyParamBounds
    // by their Debug string. The thing to keep in mind is that we don't really
    // care what the final order is - we're synthesizing an impl or bound
    // ourselves, so any order can be considered equally valid. By sorting the
    // predicates and bounds, however, we ensure that for a given codebase, all
    // auto-trait impls always render in exactly the same way.
    //
    // Using the Debug impementation for sorting prevents us from needing to
    // write quite a bit of almost entirely useless code (e.g. how should two
    // Types be sorted relative to each other). It also allows us to solve the
    // problem for both WherePredicates and TyParamBounds at the same time. This
    // approach is probably somewhat slower, but the small number of items
    // involved (impls rarely have more than a few bounds) means that it
    // shouldn't matter in practice.
    fn unstable_debug_sort<T: Debug>(&self, vec: &mut Vec<T>) {
        vec.sort_unstable_by(|first, second| {
            format!("{:?}", first).cmp(&format!("{:?}", second))
        });
    }

    fn is_fn_ty(&self, tcx: &TyCtxt, ty: &Type) -> bool {
        match &ty {
            &&Type::ResolvedPath { ref did, .. } => {
                *did == tcx.require_lang_item(lang_items::FnTraitLangItem)
                    || *did == tcx.require_lang_item(lang_items::FnMutTraitLangItem)
                    || *did == tcx.require_lang_item(lang_items::FnOnceTraitLangItem)
            }
            _ => false,
        }
    }

    // This is an ugly hack, but it's the simplest way to handle synthetic impls without greatly
    // refactoring either librustdoc or librustc. In particular, allowing new DefIds to be
    // registered after the AST is constructed would require storing the defid mapping in a
    // RefCell, decreasing the performance for normal compilation for very little gain.
    //
    // Instead, we construct 'fake' def ids, which start immediately after the last DefId in
    // DefIndexAddressSpace::Low. In the Debug impl for clean::Item, we explicitly check for fake
    // def ids, as we'll end up with a panic if we use the DefId Debug impl for fake DefIds
    fn next_def_id(&self, crate_num: CrateNum) -> DefId {
        let start_def_id = {
            let next_id = if crate_num == LOCAL_CRATE {
                self.cx
                    .tcx
                    .hir
                    .definitions()
                    .def_path_table()
                    .next_id(DefIndexAddressSpace::Low)
            } else {
                self.cx
                    .cstore
                    .def_path_table(crate_num)
                    .next_id(DefIndexAddressSpace::Low)
            };

            DefId {
                krate: crate_num,
                index: next_id,
            }
        };

        let mut fake_ids = self.cx.fake_def_ids.borrow_mut();

        let def_id = fake_ids.entry(crate_num).or_insert(start_def_id).clone();
        fake_ids.insert(
            crate_num,
            DefId {
                krate: crate_num,
                index: DefIndex::from_array_index(
                    def_id.index.as_array_index() + 1,
                    def_id.index.address_space(),
                ),
            },
        );

        MAX_DEF_ID.with(|m| {
            m.borrow_mut()
                .entry(def_id.krate.clone())
                .or_insert(start_def_id);
        });

        self.cx.all_fake_def_ids.borrow_mut().insert(def_id);

        def_id.clone()
    }
}

// Replaces all ReVars in a type with ty::Region's, using the provided map
struct RegionReplacer<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    vid_to_region: &'a FxHashMap<ty::RegionVid, ty::Region<'tcx>>,
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
}

impl<'a, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for RegionReplacer<'a, 'gcx, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> {
        self.tcx
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        (match r {
            &ty::ReVar(vid) => self.vid_to_region.get(&vid).cloned(),
            _ => None,
        }).unwrap_or_else(|| r.super_fold_with(self))
    }
}
