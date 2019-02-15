use rustc::hir;
use rustc::traits::auto_trait as auto;
use rustc::ty::{self, TypeFoldable};
use std::fmt::Debug;

use self::def_ctor::{get_def_from_def_id, get_def_from_node_id};

use super::*;

pub struct AutoTraitFinder<'a, 'tcx: 'a, 'rcx: 'a> {
    pub cx: &'a core::DocContext<'a, 'tcx, 'rcx>,
    pub f: auto::AutoTraitFinder<'a, 'tcx>,
}

impl<'a, 'tcx, 'rcx> AutoTraitFinder<'a, 'tcx, 'rcx> {
    pub fn new(cx: &'a core::DocContext<'a, 'tcx, 'rcx>) -> Self {
        let f = auto::AutoTraitFinder::new(&cx.tcx);

        AutoTraitFinder { cx, f }
    }

    pub fn get_with_def_id(&self, def_id: DefId) -> Vec<Item> {
        get_def_from_def_id(&self.cx, def_id, &|def_ctor| {
            self.get_auto_trait_impls(def_id, &def_ctor, None)
        })
    }

    pub fn get_with_node_id(&self, id: ast::NodeId, name: String) -> Vec<Item> {
        get_def_from_node_id(&self.cx, id, name, &|def_ctor, name| {
            let did = self.cx.tcx.hir().local_def_id(id);
            self.get_auto_trait_impls(did, &def_ctor, Some(name))
        })
    }

    pub fn get_auto_trait_impls<F>(
        &self,
        def_id: DefId,
        def_ctor: &F,
        name: Option<String>,
    ) -> Vec<Item>
    where F: Fn(DefId) -> Def {
        if self.cx
            .tcx
            .get_attrs(def_id)
            .lists("doc")
            .has_word("hidden")
        {
            debug!(
                "get_auto_trait_impls(def_id={:?}, def_ctor=...): item has doc('hidden'), \
                 aborting",
                def_id
            );
            return Vec::new();
        }

        let tcx = self.cx.tcx;
        let generics = self.cx.tcx.generics_of(def_id);

        debug!(
            "get_auto_trait_impls(def_id={:?}, def_ctor=..., generics={:?}",
            def_id, generics
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
                name,
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

    fn get_auto_trait_impl_for<F>(
        &self,
        def_id: DefId,
        name: Option<String>,
        generics: ty::Generics,
        def_ctor: &F,
        trait_def_id: DefId,
    ) -> Option<Item>
    where F: Fn(DefId) -> Def {
        if !self.cx
            .generated_synthetics
            .borrow_mut()
            .insert((def_id, trait_def_id))
        {
            debug!(
                "get_auto_trait_impl_for(def_id={:?}, generics={:?}, def_ctor=..., \
                 trait_def_id={:?}): already generated, aborting",
                def_id, generics, trait_def_id
            );
            return None;
        }

        let result = self.find_auto_trait_generics(def_id, trait_def_id, &generics);

        if result.is_auto() {
            let trait_ = hir::TraitRef {
                path: get_path_for_type(self.cx.tcx, trait_def_id, hir::def::Def::Trait),
                ref_id: ast::DUMMY_NODE_ID,
                hir_ref_id: hir::DUMMY_HIR_ID,
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
            let real_name = name.map(|name| Ident::from_str(&name));
            let ty = self.cx.get_real_ty(def_id, def_ctor, &real_name, &generics);

            return Some(Item {
                source: Span::empty(),
                name: None,
                attrs: Default::default(),
                visibility: None,
                def_id: self.cx.next_def_id(def_id.krate),
                stability: None,
                deprecation: None,
                inner: ImplItem(Impl {
                    unsafety: hir::Unsafety::Normal,
                    generics: new_generics,
                    provided_trait_methods: Default::default(),
                    trait_: Some(trait_.clean(self.cx)),
                    for_: ty.clean(self.cx),
                    items: Vec::new(),
                    polarity,
                    synthetic: true,
                    blanket_impl: None,
                }),
            });
        }
        None
    }

    fn find_auto_trait_generics(
        &self,
        did: DefId,
        trait_did: DefId,
        generics: &ty::Generics,
    ) -> AutoTraitResult {
        match self.f.find_auto_trait_generics(did, trait_did, generics,
                |infcx, mut info| {
                    let region_data = info.region_data;
                    let names_map =
                        info.names_map
                            .drain()
                            .map(|name| (name.clone(), Lifetime(name)))
                            .collect();
                    let lifetime_predicates =
                        self.handle_lifetimes(&region_data, &names_map);
                    let new_generics = self.param_env_to_generics(
                        infcx.tcx,
                        did,
                        info.full_user_env,
                        generics.clone(),
                        lifetime_predicates,
                        info.vid_to_region,
                    );

                    debug!(
                        "find_auto_trait_generics(did={:?}, trait_did={:?}, generics={:?}): \
                         finished with {:?}",
                        did, trait_did, generics, new_generics
                    );

                    new_generics
                }) {
            auto::AutoTraitResult::ExplicitImpl => AutoTraitResult::ExplicitImpl,
            auto::AutoTraitResult::NegativeImpl => AutoTraitResult::NegativeImpl,
            auto::AutoTraitResult::PositiveImpl(res) => AutoTraitResult::PositiveImpl(res),
        }
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
            &ty::ReEarlyBound(r) => Some(r.name.to_string()),
            _ => None,
        }
    }

    // This method calculates two things: Lifetime constraints of the form 'a: 'b,
    // and region constraints of the form ReVar: 'a
    //
    // This is essentially a simplified version of lexical_region_resolve. However,
    // handle_lifetimes determines what *needs be* true in order for an impl to hold.
    // lexical_region_resolve, along with much of the rest of the compiler, is concerned
    // with determining if a given set up constraints/predicates *are* met, given some
    // starting conditions (e.g., user-provided code). For this reason, it's easier
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
        let mut finished: FxHashMap<_, Vec<_>> = Default::default();

        let mut vid_map: FxHashMap<RegionTarget, RegionDeps> = Default::default();

        // Flattening is done in two parts. First, we insert all of the constraints
        // into a map. Each RegionTarget (either a RegionVid or a Region) maps
        // to its smaller and larger regions. Note that 'larger' regions correspond
        // to sub-regions in Rust code (e.g., in 'a: 'b, 'a is the larger region).
        for constraint in regions.constraints.keys() {
            match constraint {
                &Constraint::VarSubVar(r1, r2) => {
                    {
                        let deps1 = vid_map
                            .entry(RegionTarget::RegionVid(r1))
                            .or_default();
                        deps1.larger.insert(RegionTarget::RegionVid(r2));
                    }

                    let deps2 = vid_map
                        .entry(RegionTarget::RegionVid(r2))
                        .or_default();
                    deps2.smaller.insert(RegionTarget::RegionVid(r1));
                }
                &Constraint::RegSubVar(region, vid) => {
                    let deps = vid_map
                        .entry(RegionTarget::RegionVid(vid))
                        .or_default();
                    deps.smaller.insert(RegionTarget::Region(region));
                }
                &Constraint::VarSubReg(vid, region) => {
                    let deps = vid_map
                        .entry(RegionTarget::RegionVid(vid))
                        .or_default();
                    deps.larger.insert(RegionTarget::Region(region));
                }
                &Constraint::RegSubReg(r1, r2) => {
                    // The constraint is already in the form that we want, so we're done with it
                    // Desired order is 'larger, smaller', so flip then
                    if self.region_name(r1) != self.region_name(r2) {
                        finished
                            .entry(self.region_name(r2).expect("no region_name found"))
                            .or_default()
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
                                    .entry(self.region_name(r2).expect("no region name found"))
                                    .or_default()
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
                let bounds: FxHashSet<GenericBound> = finished.get(name).unwrap_or(&empty).iter()
                    .map(|region| GenericBound::Outlives(self.get_lifetime(region, names_map)))
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
    ) -> FxHashSet<GenericParamDef> {
        pred.walk_tys()
            .flat_map(|t| {
                let mut regions = FxHashSet::default();
                tcx.collect_regions(&t, &mut regions);

                regions.into_iter().flat_map(|r| {
                    match r {
                        // We only care about late bound regions, as we need to add them
                        // to the 'for<>' section
                        &ty::ReLateBound(_, ty::BoundRegion::BrNamed(_, name)) => {
                            Some(GenericParamDef {
                                name: name.to_string(),
                                kind: GenericParamDefKind::Lifetime,
                            })
                        },
                        &ty::ReVar(_) | &ty::ReEarlyBound(_) | &ty::ReStatic => None,
                        _ => panic!("Unexpected region type {:?}", r),
                    }
                })
            })
            .collect()
    }

    fn make_final_bounds<'b, 'c, 'cx>(
        &self,
        ty_to_bounds: FxHashMap<Type, FxHashSet<GenericBound>>,
        ty_to_fn: FxHashMap<Type, (Option<PolyTrait>, Option<Type>)>,
        lifetime_to_bounds: FxHashMap<Lifetime, FxHashSet<GenericBound>>,
    ) -> Vec<WherePredicate> {
        ty_to_bounds
            .into_iter()
            .flat_map(|(ty, mut bounds)| {
                if let Some(data) = ty_to_fn.get(&ty) {
                    let (poly_trait, output) =
                        (data.0.as_ref().expect("as_ref failed").clone(), data.1.as_ref().cloned());
                    let new_ty = match &poly_trait.trait_ {
                        &Type::ResolvedPath {
                            ref path,
                            ref typarams,
                            ref did,
                            ref is_generic,
                        } => {
                            let mut new_path = path.clone();
                            let last_segment = new_path.segments.pop()
                                                                .expect("segments were empty");

                            let (old_input, old_output) = match last_segment.args {
                                GenericArgs::AngleBracketed { types, .. } => (types, None),
                                GenericArgs::Parenthesized { inputs, output, .. } => {
                                    (inputs, output)
                                }
                            };

                            if old_output.is_some() && old_output != output {
                                panic!(
                                    "Output mismatch for {:?} {:?} {:?}",
                                    ty, old_output, data.1
                                );
                            }

                            let new_params = GenericArgs::Parenthesized {
                                inputs: old_input,
                                output,
                            };

                            new_path.segments.push(PathSegment {
                                name: last_segment.name,
                                args: new_params,
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
                    bounds.insert(GenericBound::TraitBound(
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
                        self.sort_where_bounds(&mut bounds_vec);
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
    // * Multiple bounds for the same type are coalesced into one: e.g., 'T: Copy', 'T: Debug'
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

        // The `Sized` trait must be handled specially, since we only display it when
        // it is *not* required (i.e., '?Sized')
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

        let mut has_sized = FxHashSet::default();
        let mut ty_to_bounds: FxHashMap<_, FxHashSet<_>> = Default::default();
        let mut lifetime_to_bounds: FxHashMap<_, FxHashSet<_>> = Default::default();
        let mut ty_to_traits: FxHashMap<Type, FxHashSet<Type>> = Default::default();

        let mut ty_to_fn: FxHashMap<Type, (Option<PolyTrait>, Option<Type>)> = Default::default();

        for (orig_p, p) in clean_where_predicates {
            if p.is_none() {
                continue;
            }
            let p = p.unwrap();
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
                    let mut b = bounds.pop().expect("bounds were empty");

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
                            &mut GenericBound::TraitBound(ref mut p, _) => {
                                // Insert regions into the for_generics hash map first, to ensure
                                // that we don't end up with duplicate bounds (e.g., for<'b, 'b>)
                                for_generics.extend(p.generic_params.clone());
                                p.generic_params = for_generics.into_iter().collect();
                                self.is_fn_ty(&tcx, &p.trait_)
                            }
                            _ => false,
                        };

                        let poly_trait = b.get_poly_trait().expect("Cannot get poly trait");

                        if is_fn {
                            ty_to_fn
                                .entry(ty.clone())
                                .and_modify(|e| *e = (Some(poly_trait.clone()), e.1.clone()))
                                .or_insert(((Some(poly_trait.clone())), None));

                            ty_to_bounds
                                .entry(ty.clone())
                                .or_default();
                        } else {
                            ty_to_bounds
                                .entry(ty.clone())
                                .or_default()
                                .insert(b.clone());
                        }
                    }
                }
                WherePredicate::RegionPredicate { lifetime, bounds } => {
                    lifetime_to_bounds
                        .entry(lifetime)
                        .or_default()
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
                                        let args =
                                            &mut new_trait_path.segments
                                                .last_mut()
                                                .expect("segments were empty")
                                                .args;

                                        match args {
                                            // Convert somethiung like '<T as Iterator::Item> = u8'
                                            // to 'T: Iterator<Item=u8>'
                                            &mut GenericArgs::AngleBracketed {
                                                ref mut bindings,
                                                ..
                                            } => {
                                                bindings.push(TypeBinding {
                                                    name: left_name.clone(),
                                                    ty: rhs,
                                                });
                                            }
                                            &mut GenericArgs::Parenthesized { .. } => {
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
                                        .or_default();

                                    bounds.insert(GenericBound::TraitBound(
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

                                    // Remove any existing 'plain' bound (e.g., 'T: Iterator`) so
                                    // that we don't see a
                                    // duplicate bound like `T: Iterator + Iterator<Item=u8>`
                                    // on the docs page.
                                    bounds.remove(&GenericBound::TraitBound(
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
                                        .or_default()
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

        for param in generic_params.iter_mut() {
            match param.kind {
                GenericParamDefKind::Type { ref mut default, ref mut bounds, .. } => {
                    // We never want something like `impl<T=Foo>`.
                    default.take();
                    let generic_ty = Type::Generic(param.name.clone());
                    if !has_sized.contains(&generic_ty) {
                        bounds.insert(0, GenericBound::maybe_sized(self.cx));
                    }
                }
                GenericParamDefKind::Lifetime => {}
                GenericParamDefKind::Const { .. } => {}
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
    fn sort_where_bounds(&self, mut bounds: &mut Vec<GenericBound>) {
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
    // Obviously, it's extremely undesirable for documentation rendering
    // to be depndent on the platform it's run on. Apart from being confusing
    // to end users, it makes writing tests much more difficult, as predicates
    // can appear in any order in the final result.
    //
    // To solve this problem, we sort WherePredicates and GenericBounds
    // by their Debug string. The thing to keep in mind is that we don't really
    // care what the final order is - we're synthesizing an impl or bound
    // ourselves, so any order can be considered equally valid. By sorting the
    // predicates and bounds, however, we ensure that for a given codebase, all
    // auto-trait impls always render in exactly the same way.
    //
    // Using the Debug implementation for sorting prevents us from needing to
    // write quite a bit of almost entirely useless code (e.g., how should two
    // Types be sorted relative to each other). It also allows us to solve the
    // problem for both WherePredicates and GenericBounds at the same time. This
    // approach is probably somewhat slower, but the small number of items
    // involved (impls rarely have more than a few bounds) means that it
    // shouldn't matter in practice.
    fn unstable_debug_sort<T: Debug>(&self, vec: &mut Vec<T>) {
        vec.sort_by_cached_key(|x| format!("{:?}", x))
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
