use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::lang_items::LangItem;
use rustc_middle::ty::{self, Region, RegionVid, TypeFoldable};
use rustc_trait_selection::traits::auto_trait::{self, AutoTraitResult};
use thin_vec::ThinVec;

use std::fmt::Debug;

use super::*;

#[derive(Eq, PartialEq, Hash, Copy, Clone, Debug)]
enum RegionTarget<'tcx> {
    Region(Region<'tcx>),
    RegionVid(RegionVid),
}

#[derive(Default, Debug, Clone)]
struct RegionDeps<'tcx> {
    larger: FxHashSet<RegionTarget<'tcx>>,
    smaller: FxHashSet<RegionTarget<'tcx>>,
}

pub(crate) struct AutoTraitFinder<'a, 'tcx> {
    pub(crate) cx: &'a mut core::DocContext<'tcx>,
}

impl<'a, 'tcx> AutoTraitFinder<'a, 'tcx>
where
    'tcx: 'a, // should be an implied bound; rustc bug #98852.
{
    pub(crate) fn new(cx: &'a mut core::DocContext<'tcx>) -> Self {
        AutoTraitFinder { cx }
    }

    fn generate_for_trait(
        &mut self,
        ty: Ty<'tcx>,
        trait_def_id: DefId,
        param_env: ty::ParamEnv<'tcx>,
        item_def_id: DefId,
        f: &auto_trait::AutoTraitFinder<'tcx>,
        // If this is set, show only negative trait implementations, not positive ones.
        discard_positive_impl: bool,
    ) -> Option<Item> {
        let tcx = self.cx.tcx;
        let trait_ref = ty::Binder::dummy(ty::TraitRef::new(tcx, trait_def_id, [ty]));
        if !self.cx.generated_synthetics.insert((ty, trait_def_id)) {
            debug!("get_auto_trait_impl_for({:?}): already generated, aborting", trait_ref);
            return None;
        }

        let result = f.find_auto_trait_generics(ty, param_env, trait_def_id, |info| {
            let region_data = info.region_data;

            let names_map = tcx
                .generics_of(item_def_id)
                .params
                .iter()
                .filter_map(|param| match param.kind {
                    ty::GenericParamDefKind::Lifetime => Some(param.name),
                    _ => None,
                })
                .map(|name| (name, Lifetime(name)))
                .collect();
            let lifetime_predicates = Self::handle_lifetimes(&region_data, &names_map);
            let new_generics = self.param_env_to_generics(
                item_def_id,
                info.full_user_env,
                lifetime_predicates,
                info.vid_to_region,
            );

            debug!(
                "find_auto_trait_generics(item_def_id={:?}, trait_def_id={:?}): \
                    finished with {:?}",
                item_def_id, trait_def_id, new_generics
            );

            new_generics
        });

        let polarity;
        let new_generics = match result {
            AutoTraitResult::PositiveImpl(new_generics) => {
                polarity = ty::ImplPolarity::Positive;
                if discard_positive_impl {
                    return None;
                }
                new_generics
            }
            AutoTraitResult::NegativeImpl => {
                polarity = ty::ImplPolarity::Negative;

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
                let raw_generics = clean_ty_generics(
                    self.cx,
                    tcx.generics_of(item_def_id),
                    ty::GenericPredicates::default(),
                );
                let params = raw_generics.params;

                Generics { params, where_predicates: ThinVec::new() }
            }
            AutoTraitResult::ExplicitImpl => return None,
        };

        Some(Item {
            name: None,
            attrs: Default::default(),
            item_id: ItemId::Auto { trait_: trait_def_id, for_: item_def_id },
            kind: Box::new(ImplItem(Box::new(Impl {
                unsafety: hir::Unsafety::Normal,
                generics: new_generics,
                trait_: Some(clean_trait_ref_with_bindings(self.cx, trait_ref, ThinVec::new())),
                for_: clean_middle_ty(ty::Binder::dummy(ty), self.cx, None, None),
                items: Vec::new(),
                polarity,
                kind: ImplKind::Auto,
            }))),
            cfg: None,
            inline_stmt_id: None,
        })
    }

    pub(crate) fn get_auto_trait_impls(&mut self, item_def_id: DefId) -> Vec<Item> {
        let tcx = self.cx.tcx;
        let param_env = tcx.param_env(item_def_id);
        let ty = tcx.type_of(item_def_id).subst_identity();
        let f = auto_trait::AutoTraitFinder::new(tcx);

        debug!("get_auto_trait_impls({:?})", ty);
        let auto_traits: Vec<_> = self.cx.auto_traits.to_vec();
        let mut auto_traits: Vec<Item> = auto_traits
            .into_iter()
            .filter_map(|trait_def_id| {
                self.generate_for_trait(ty, trait_def_id, param_env, item_def_id, &f, false)
            })
            .collect();
        // We are only interested in case the type *doesn't* implement the Sized trait.
        if !ty.is_sized(tcx, param_env) {
            // In case `#![no_core]` is used, `sized_trait` returns nothing.
            if let Some(item) = tcx.lang_items().sized_trait().and_then(|sized_trait_did| {
                self.generate_for_trait(ty, sized_trait_did, param_env, item_def_id, &f, true)
            }) {
                auto_traits.push(item);
            }
        }
        auto_traits
    }

    fn get_lifetime(region: Region<'_>, names_map: &FxHashMap<Symbol, Lifetime>) -> Lifetime {
        region_name(region)
            .map(|name| {
                names_map.get(&name).unwrap_or_else(|| {
                    panic!("Missing lifetime with name {:?} for {:?}", name.as_str(), region)
                })
            })
            .unwrap_or(&Lifetime::statik())
            .clone()
    }

    /// This method calculates two things: Lifetime constraints of the form `'a: 'b`,
    /// and region constraints of the form `RegionVid: 'a`
    ///
    /// This is essentially a simplified version of lexical_region_resolve. However,
    /// handle_lifetimes determines what *needs be* true in order for an impl to hold.
    /// lexical_region_resolve, along with much of the rest of the compiler, is concerned
    /// with determining if a given set up constraints/predicates *are* met, given some
    /// starting conditions (e.g., user-provided code). For this reason, it's easier
    /// to perform the calculations we need on our own, rather than trying to make
    /// existing inference/solver code do what we want.
    fn handle_lifetimes<'cx>(
        regions: &RegionConstraintData<'cx>,
        names_map: &FxHashMap<Symbol, Lifetime>,
    ) -> ThinVec<WherePredicate> {
        // Our goal is to 'flatten' the list of constraints by eliminating
        // all intermediate RegionVids. At the end, all constraints should
        // be between Regions (aka region variables). This gives us the information
        // we need to create the Generics.
        let mut finished: FxHashMap<_, Vec<_>> = Default::default();

        let mut vid_map: FxHashMap<RegionTarget<'_>, RegionDeps<'_>> = Default::default();

        // Flattening is done in two parts. First, we insert all of the constraints
        // into a map. Each RegionTarget (either a RegionVid or a Region) maps
        // to its smaller and larger regions. Note that 'larger' regions correspond
        // to sub-regions in Rust code (e.g., in 'a: 'b, 'a is the larger region).
        for constraint in regions.constraints.keys() {
            match *constraint {
                Constraint::VarSubVar(r1, r2) => {
                    {
                        let deps1 = vid_map.entry(RegionTarget::RegionVid(r1)).or_default();
                        deps1.larger.insert(RegionTarget::RegionVid(r2));
                    }

                    let deps2 = vid_map.entry(RegionTarget::RegionVid(r2)).or_default();
                    deps2.smaller.insert(RegionTarget::RegionVid(r1));
                }
                Constraint::RegSubVar(region, vid) => {
                    let deps = vid_map.entry(RegionTarget::RegionVid(vid)).or_default();
                    deps.smaller.insert(RegionTarget::Region(region));
                }
                Constraint::VarSubReg(vid, region) => {
                    let deps = vid_map.entry(RegionTarget::RegionVid(vid)).or_default();
                    deps.larger.insert(RegionTarget::Region(region));
                }
                Constraint::RegSubReg(r1, r2) => {
                    // The constraint is already in the form that we want, so we're done with it
                    // Desired order is 'larger, smaller', so flip then
                    if region_name(r1) != region_name(r2) {
                        finished
                            .entry(region_name(r2).expect("no region_name found"))
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
            let target = *vid_map.keys().next().expect("Keys somehow empty");
            let deps = vid_map.remove(&target).expect("Entry somehow missing");

            for smaller in deps.smaller.iter() {
                for larger in deps.larger.iter() {
                    match (smaller, larger) {
                        (&RegionTarget::Region(r1), &RegionTarget::Region(r2)) => {
                            if region_name(r1) != region_name(r2) {
                                finished
                                    .entry(region_name(r2).expect("no region name found"))
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
                let bounds: FxHashSet<GenericBound> = finished
                    .get(name)
                    .unwrap_or(&empty)
                    .iter()
                    .map(|region| GenericBound::Outlives(Self::get_lifetime(*region, names_map)))
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

    fn extract_for_generics(&self, pred: ty::Predicate<'tcx>) -> FxHashSet<GenericParamDef> {
        let bound_predicate = pred.kind();
        let tcx = self.cx.tcx;
        let regions = match bound_predicate.skip_binder() {
            ty::PredicateKind::Clause(ty::Clause::Trait(poly_trait_pred)) => {
                tcx.collect_referenced_late_bound_regions(&bound_predicate.rebind(poly_trait_pred))
            }
            ty::PredicateKind::Clause(ty::Clause::Projection(poly_proj_pred)) => {
                tcx.collect_referenced_late_bound_regions(&bound_predicate.rebind(poly_proj_pred))
            }
            _ => return FxHashSet::default(),
        };

        regions
            .into_iter()
            .filter_map(|br| {
                match br {
                    // We only care about named late bound regions, as we need to add them
                    // to the 'for<>' section
                    ty::BrNamed(_, name) => Some(GenericParamDef::lifetime(name)),
                    _ => None,
                }
            })
            .collect()
    }

    fn make_final_bounds(
        &self,
        ty_to_bounds: FxHashMap<Type, FxHashSet<GenericBound>>,
        ty_to_fn: FxHashMap<Type, (PolyTrait, Option<Type>)>,
        lifetime_to_bounds: FxHashMap<Lifetime, FxHashSet<GenericBound>>,
    ) -> Vec<WherePredicate> {
        ty_to_bounds
            .into_iter()
            .flat_map(|(ty, mut bounds)| {
                if let Some((ref poly_trait, ref output)) = ty_to_fn.get(&ty) {
                    let mut new_path = poly_trait.trait_.clone();
                    let last_segment = new_path.segments.pop().expect("segments were empty");

                    let (old_input, old_output) = match last_segment.args {
                        GenericArgs::AngleBracketed { args, .. } => {
                            let types = args
                                .iter()
                                .filter_map(|arg| match arg {
                                    GenericArg::Type(ty) => Some(ty.clone()),
                                    _ => None,
                                })
                                .collect();
                            (types, None)
                        }
                        GenericArgs::Parenthesized { inputs, output } => (inputs, output),
                    };

                    let output = output.as_ref().cloned().map(Box::new);
                    if old_output.is_some() && old_output != output {
                        panic!("Output mismatch for {:?} {:?} {:?}", ty, old_output, output);
                    }

                    let new_params = GenericArgs::Parenthesized { inputs: old_input, output };

                    new_path
                        .segments
                        .push(PathSegment { name: last_segment.name, args: new_params });

                    bounds.insert(GenericBound::TraitBound(
                        PolyTrait {
                            trait_: new_path,
                            generic_params: poly_trait.generic_params.clone(),
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
                    bound_params: Vec::new(),
                })
            })
            .chain(lifetime_to_bounds.into_iter().filter(|(_, bounds)| !bounds.is_empty()).map(
                |(lifetime, bounds)| {
                    let mut bounds_vec = bounds.into_iter().collect();
                    self.sort_where_bounds(&mut bounds_vec);
                    WherePredicate::RegionPredicate { lifetime, bounds: bounds_vec }
                },
            ))
            .collect()
    }

    /// Converts the calculated `ParamEnv` and lifetime information to a [`clean::Generics`](Generics), suitable for
    /// display on the docs page. Cleaning the `Predicates` produces sub-optimal [`WherePredicate`]s,
    /// so we fix them up:
    ///
    /// * Multiple bounds for the same type are coalesced into one: e.g., `T: Copy`, `T: Debug`
    /// becomes `T: Copy + Debug`
    /// * `Fn` bounds are handled specially - instead of leaving it as `T: Fn(), <T as Fn::Output> =
    /// K`, we use the dedicated syntax `T: Fn() -> K`
    /// * We explicitly add a `?Sized` bound if we didn't find any `Sized` predicates for a type
    fn param_env_to_generics(
        &mut self,
        item_def_id: DefId,
        param_env: ty::ParamEnv<'tcx>,
        mut existing_predicates: ThinVec<WherePredicate>,
        vid_to_region: FxHashMap<ty::RegionVid, ty::Region<'tcx>>,
    ) -> Generics {
        debug!(
            "param_env_to_generics(item_def_id={:?}, param_env={:?}, \
             existing_predicates={:?})",
            item_def_id, param_env, existing_predicates
        );

        let tcx = self.cx.tcx;

        // The `Sized` trait must be handled specially, since we only display it when
        // it is *not* required (i.e., '?Sized')
        let sized_trait = tcx.require_lang_item(LangItem::Sized, None);

        let mut replacer = RegionReplacer { vid_to_region: &vid_to_region, tcx };

        let orig_bounds: FxHashSet<_> = tcx.param_env(item_def_id).caller_bounds().iter().collect();
        let clean_where_predicates = param_env
            .caller_bounds()
            .iter()
            .filter(|p| {
                !orig_bounds.contains(p)
                    || match p.kind().skip_binder() {
                        ty::PredicateKind::Clause(ty::Clause::Trait(pred)) => {
                            pred.def_id() == sized_trait
                        }
                        _ => false,
                    }
            })
            .map(|p| p.fold_with(&mut replacer));

        let raw_generics = clean_ty_generics(
            self.cx,
            tcx.generics_of(item_def_id),
            tcx.explicit_predicates_of(item_def_id),
        );
        let mut generic_params = raw_generics.params;

        debug!("param_env_to_generics({:?}): generic_params={:?}", item_def_id, generic_params);

        let mut has_sized = FxHashSet::default();
        let mut ty_to_bounds: FxHashMap<_, FxHashSet<_>> = Default::default();
        let mut lifetime_to_bounds: FxHashMap<_, FxHashSet<_>> = Default::default();
        let mut ty_to_traits: FxHashMap<Type, FxHashSet<Path>> = Default::default();

        let mut ty_to_fn: FxHashMap<Type, (PolyTrait, Option<Type>)> = Default::default();

        // FIXME: This code shares much of the logic found in `clean_ty_generics` and
        //        `simplify::where_clause`. Consider deduplicating it to avoid diverging
        //        implementations.
        //        Further, the code below does not merge (partially re-sugared) bounds like
        //        `Tr<A = T>` & `Tr<B = U>` and it does not render higher-ranked parameters
        //        originating from equality predicates.
        for p in clean_where_predicates {
            let (orig_p, p) = (p, clean_predicate(p, self.cx));
            if p.is_none() {
                continue;
            }
            let p = p.unwrap();
            match p {
                WherePredicate::BoundPredicate { ty, mut bounds, .. } => {
                    // Writing a projection trait bound of the form
                    // <T as Trait>::Name : ?Sized
                    // is illegal, because ?Sized bounds can only
                    // be written in the (here, nonexistent) definition
                    // of the type.
                    // Therefore, we make sure that we never add a ?Sized
                    // bound for projections
                    if let Type::QPath { .. } = ty {
                        has_sized.insert(ty.clone());
                    }

                    if bounds.is_empty() {
                        continue;
                    }

                    let mut for_generics = self.extract_for_generics(orig_p);

                    assert!(bounds.len() == 1);
                    let mut b = bounds.pop().expect("bounds were empty");

                    if b.is_sized_bound(self.cx) {
                        has_sized.insert(ty.clone());
                    } else if !b
                        .get_trait_path()
                        .and_then(|trait_| {
                            ty_to_traits
                                .get(&ty)
                                .map(|bounds| bounds.contains(&strip_path_generics(trait_)))
                        })
                        .unwrap_or(false)
                    {
                        // If we've already added a projection bound for the same type, don't add
                        // this, as it would be a duplicate

                        // Handle any 'Fn/FnOnce/FnMut' bounds specially,
                        // as we want to combine them with any 'Output' qpaths
                        // later

                        let is_fn = match b {
                            GenericBound::TraitBound(ref mut p, _) => {
                                // Insert regions into the for_generics hash map first, to ensure
                                // that we don't end up with duplicate bounds (e.g., for<'b, 'b>)
                                for_generics.extend(p.generic_params.drain(..));
                                p.generic_params.extend(for_generics);
                                self.is_fn_trait(&p.trait_)
                            }
                            _ => false,
                        };

                        let poly_trait = b.get_poly_trait().expect("Cannot get poly trait");

                        if is_fn {
                            ty_to_fn
                                .entry(ty.clone())
                                .and_modify(|e| *e = (poly_trait.clone(), e.1.clone()))
                                .or_insert(((poly_trait.clone()), None));

                            ty_to_bounds.entry(ty.clone()).or_default();
                        } else {
                            ty_to_bounds.entry(ty.clone()).or_default().insert(b.clone());
                        }
                    }
                }
                WherePredicate::RegionPredicate { lifetime, bounds } => {
                    lifetime_to_bounds.entry(lifetime).or_default().extend(bounds);
                }
                WherePredicate::EqPredicate { lhs, rhs, bound_params } => {
                    match *lhs {
                        Type::QPath(box QPathData {
                            ref assoc,
                            ref self_type,
                            trait_: Some(ref trait_),
                            ..
                        }) => {
                            let ty = &*self_type;
                            let mut new_trait = trait_.clone();

                            if self.is_fn_trait(trait_) && assoc.name == sym::Output {
                                ty_to_fn
                                    .entry(ty.clone())
                                    .and_modify(|e| {
                                        *e = (e.0.clone(), Some(rhs.ty().unwrap().clone()))
                                    })
                                    .or_insert((
                                        PolyTrait {
                                            trait_: trait_.clone(),
                                            generic_params: Vec::new(),
                                        },
                                        Some(rhs.ty().unwrap().clone()),
                                    ));
                                continue;
                            }

                            let args = &mut new_trait
                                .segments
                                .last_mut()
                                .expect("segments were empty")
                                .args;

                            match args {
                                // Convert something like '<T as Iterator::Item> = u8'
                                // to 'T: Iterator<Item=u8>'
                                GenericArgs::AngleBracketed { ref mut bindings, .. } => {
                                    bindings.push(TypeBinding {
                                        assoc: assoc.clone(),
                                        kind: TypeBindingKind::Equality { term: *rhs },
                                    });
                                }
                                GenericArgs::Parenthesized { .. } => {
                                    existing_predicates.push(WherePredicate::EqPredicate {
                                        lhs: lhs.clone(),
                                        rhs,
                                        bound_params,
                                    });
                                    continue; // If something other than a Fn ends up
                                    // with parentheses, leave it alone
                                }
                            }

                            let bounds = ty_to_bounds.entry(ty.clone()).or_default();

                            bounds.insert(GenericBound::TraitBound(
                                PolyTrait { trait_: new_trait, generic_params: Vec::new() },
                                hir::TraitBoundModifier::None,
                            ));

                            // Remove any existing 'plain' bound (e.g., 'T: Iterator`) so
                            // that we don't see a
                            // duplicate bound like `T: Iterator + Iterator<Item=u8>`
                            // on the docs page.
                            bounds.remove(&GenericBound::TraitBound(
                                PolyTrait { trait_: trait_.clone(), generic_params: Vec::new() },
                                hir::TraitBoundModifier::None,
                            ));
                            // Avoid creating any new duplicate bounds later in the outer
                            // loop
                            ty_to_traits.entry(ty.clone()).or_default().insert(trait_.clone());
                        }
                        _ => panic!("Unexpected LHS {:?} for {:?}", lhs, item_def_id),
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
                    let generic_ty = Type::Generic(param.name);
                    if !has_sized.contains(&generic_ty) {
                        bounds.insert(0, GenericBound::maybe_sized(self.cx));
                    }
                }
                GenericParamDefKind::Lifetime { .. } => {}
                GenericParamDefKind::Const { ref mut default, .. } => {
                    // We never want something like `impl<const N: usize = 10>`
                    default.take();
                }
            }
        }

        self.sort_where_predicates(&mut existing_predicates);

        Generics { params: generic_params, where_predicates: existing_predicates }
    }

    /// Ensure that the predicates are in a consistent order. The precise
    /// ordering doesn't actually matter, but it's important that
    /// a given set of predicates always appears in the same order -
    /// both for visual consistency between 'rustdoc' runs, and to
    /// make writing tests much easier
    #[inline]
    fn sort_where_predicates(&self, predicates: &mut [WherePredicate]) {
        // We should never have identical bounds - and if we do,
        // they're visually identical as well. Therefore, using
        // an unstable sort is fine.
        self.unstable_debug_sort(predicates);
    }

    /// Ensure that the bounds are in a consistent order. The precise
    /// ordering doesn't actually matter, but it's important that
    /// a given set of bounds always appears in the same order -
    /// both for visual consistency between 'rustdoc' runs, and to
    /// make writing tests much easier
    #[inline]
    fn sort_where_bounds(&self, bounds: &mut Vec<GenericBound>) {
        // We should never have identical bounds - and if we do,
        // they're visually identical as well. Therefore, using
        // an unstable sort is fine.
        self.unstable_debug_sort(bounds);
    }

    /// This might look horrendously hacky, but it's actually not that bad.
    ///
    /// For performance reasons, we use several different FxHashMaps
    /// in the process of computing the final set of where predicates.
    /// However, the iteration order of a HashMap is completely unspecified.
    /// In fact, the iteration of an FxHashMap can even vary between platforms,
    /// since FxHasher has different behavior for 32-bit and 64-bit platforms.
    ///
    /// Obviously, it's extremely undesirable for documentation rendering
    /// to be dependent on the platform it's run on. Apart from being confusing
    /// to end users, it makes writing tests much more difficult, as predicates
    /// can appear in any order in the final result.
    ///
    /// To solve this problem, we sort WherePredicates and GenericBounds
    /// by their Debug string. The thing to keep in mind is that we don't really
    /// care what the final order is - we're synthesizing an impl or bound
    /// ourselves, so any order can be considered equally valid. By sorting the
    /// predicates and bounds, however, we ensure that for a given codebase, all
    /// auto-trait impls always render in exactly the same way.
    ///
    /// Using the Debug implementation for sorting prevents us from needing to
    /// write quite a bit of almost entirely useless code (e.g., how should two
    /// Types be sorted relative to each other). It also allows us to solve the
    /// problem for both WherePredicates and GenericBounds at the same time. This
    /// approach is probably somewhat slower, but the small number of items
    /// involved (impls rarely have more than a few bounds) means that it
    /// shouldn't matter in practice.
    fn unstable_debug_sort<T: Debug>(&self, vec: &mut [T]) {
        vec.sort_by_cached_key(|x| format!("{:?}", x))
    }

    fn is_fn_trait(&self, path: &Path) -> bool {
        let tcx = self.cx.tcx;
        let did = path.def_id();
        did == tcx.require_lang_item(LangItem::Fn, None)
            || did == tcx.require_lang_item(LangItem::FnMut, None)
            || did == tcx.require_lang_item(LangItem::FnOnce, None)
    }
}

fn region_name(region: Region<'_>) -> Option<Symbol> {
    match *region {
        ty::ReEarlyBound(r) => Some(r.name),
        _ => None,
    }
}

/// Replaces all [`ty::RegionVid`]s in a type with [`ty::Region`]s, using the provided map.
struct RegionReplacer<'a, 'tcx> {
    vid_to_region: &'a FxHashMap<ty::RegionVid, ty::Region<'tcx>>,
    tcx: TyCtxt<'tcx>,
}

impl<'a, 'tcx> TypeFolder<TyCtxt<'tcx>> for RegionReplacer<'a, 'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
            // These are the regions that can be seen in the AST.
            ty::ReVar(vid) => self.vid_to_region.get(&vid).cloned().unwrap_or(r),
            ty::ReEarlyBound(_) | ty::ReStatic | ty::ReLateBound(..) | ty::ReError(_) => r,
            r => bug!("unexpected region: {r:?}"),
        }
    }
}
