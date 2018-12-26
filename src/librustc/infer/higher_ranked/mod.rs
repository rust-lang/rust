//! Helper routines for higher-ranked things. See the `doc` module at
//! the end of the file for details.

use super::{CombinedSnapshot,
            InferCtxt,
            HigherRankedType,
            SubregionOrigin,
            PlaceholderMap};
use super::combine::CombineFields;
use super::region_constraints::{TaintDirections};

use ty::{self, TyCtxt, Binder, TypeFoldable};
use ty::error::TypeError;
use ty::relate::{Relate, RelateResult, TypeRelation};
use syntax_pos::Span;
use util::nodemap::{FxHashMap, FxHashSet};

pub struct HrMatchResult<U> {
    pub value: U,
}

impl<'a, 'gcx, 'tcx> CombineFields<'a, 'gcx, 'tcx> {
    pub fn higher_ranked_sub<T>(&mut self, a: &Binder<T>, b: &Binder<T>, a_is_expected: bool)
                                -> RelateResult<'tcx, Binder<T>>
        where T: Relate<'tcx>
    {
        debug!("higher_ranked_sub(a={:?}, b={:?})",
               a, b);

        // Rather than checking the subtype relationship between `a` and `b`
        // as-is, we need to do some extra work here in order to make sure
        // that function subtyping works correctly with respect to regions
        //
        // Note: this is a subtle algorithm.  For a full explanation,
        // please see the large comment at the end of the file in the (inlined) module
        // `doc`.

        // Start a snapshot so we can examine "all bindings that were
        // created as part of this type comparison".
        return self.infcx.commit_if_ok(|snapshot| {
            let span = self.trace.cause.span;

            // First, we instantiate each bound region in the supertype with a
            // fresh placeholder region.
            let (b_prime, placeholder_map) =
                self.infcx.replace_bound_vars_with_placeholders(b);

            // Next, we instantiate each bound region in the subtype
            // with a fresh region variable. These region variables --
            // but no other pre-existing region variables -- can name
            // the placeholders.
            let (a_prime, _) = self.infcx.replace_bound_vars_with_fresh_vars(
                span,
                HigherRankedType,
                a
            );

            debug!("a_prime={:?}", a_prime);
            debug!("b_prime={:?}", b_prime);

            // Compare types now that bound regions have been replaced.
            let result = self.sub(a_is_expected).relate(&a_prime, &b_prime)?;

            // Presuming type comparison succeeds, we need to check
            // that the placeholder regions do not "leak".
            self.infcx.leak_check(!a_is_expected, span, &placeholder_map, snapshot)?;

            // We are finished with the placeholder regions now so pop
            // them off.
            self.infcx.pop_placeholders(placeholder_map, snapshot);

            debug!("higher_ranked_sub: OK result={:?}", result);

            Ok(ty::Binder::bind(result))
        });
    }

    /// The value consists of a pair `(t, u)` where `t` is the
    /// *matcher* and `u` is a *value*. The idea is to find a
    /// substitution `S` such that `S(t) == b`, and then return
    /// `S(u)`. In other words, find values for the late-bound regions
    /// in `a` that can make `t == b` and then replace the LBR in `u`
    /// with those values.
    ///
    /// This routine is (as of this writing) used in trait matching,
    /// particularly projection.
    ///
    /// NB. It should not happen that there are LBR appearing in `U`
    /// that do not appear in `T`. If that happens, those regions are
    /// unconstrained, and this routine replaces them with `'static`.
    pub fn higher_ranked_match<T, U>(&mut self,
                                     a_pair: &Binder<(T, U)>,
                                     b_match: &T,
                                     a_is_expected: bool)
                                     -> RelateResult<'tcx, HrMatchResult<U>>
        where T: Relate<'tcx>,
              U: TypeFoldable<'tcx>
    {
        debug!("higher_ranked_match(a={:?}, b={:?})",
               a_pair, b_match);

        // Start a snapshot so we can examine "all bindings that were
        // created as part of this type comparison".
        return self.infcx.commit_if_ok(|snapshot| {
            // First, we instantiate each bound region in the matcher
            // with a placeholder region.
            let ((a_match, a_value), placeholder_map) =
                self.infcx.replace_bound_vars_with_placeholders(a_pair);

            debug!("higher_ranked_match: a_match={:?}", a_match);
            debug!("higher_ranked_match: placeholder_map={:?}", placeholder_map);

            // Equate types now that bound regions have been replaced.
            self.equate(a_is_expected).relate(&a_match, &b_match)?;

            // Map each placeholder region to a vector of other regions that it
            // must be equated with. (Note that this vector may include other
            // placeholder regions from `placeholder_map`.)
            let placeholder_resolution_map: FxHashMap<_, _> =
                placeholder_map
                .iter()
                .map(|(&br, &placeholder)| {
                    let tainted_regions =
                        self.infcx.tainted_regions(snapshot,
                                                   placeholder,
                                                   TaintDirections::incoming()); // [1]

                    // [1] this routine executes after the placeholder
                    // regions have been *equated* with something
                    // else, so examining the incoming edges ought to
                    // be enough to collect all constraints

                    (placeholder, (br, tainted_regions))
                })
                .collect();

            // For each placeholder region, pick a representative -- which can
            // be any region from the sets above, except for other members of
            // `placeholder_map`. There should always be a representative if things
            // are properly well-formed.
            let placeholder_representatives: FxHashMap<_, _> =
                placeholder_resolution_map
                .iter()
                .map(|(&placeholder, &(_, ref regions))| {
                    let representative =
                        regions.iter()
                               .filter(|&&r| !placeholder_resolution_map.contains_key(r))
                               .cloned()
                               .next()
                               .unwrap_or_else(|| {
                                   bug!("no representative region for `{:?}` in `{:?}`",
                                        placeholder, regions)
                               });

                    (placeholder, representative)
                })
                .collect();

            // Equate all the members of each placeholder set with the
            // representative.
            for (placeholder, &(_br, ref regions)) in &placeholder_resolution_map {
                let representative = &placeholder_representatives[placeholder];
                debug!("higher_ranked_match: \
                        placeholder={:?} representative={:?} regions={:?}",
                       placeholder, representative, regions);
                for region in regions.iter()
                                     .filter(|&r| !placeholder_resolution_map.contains_key(r))
                                     .filter(|&r| r != representative)
                {
                    let origin = SubregionOrigin::Subtype(self.trace.clone());
                    self.infcx.borrow_region_constraints()
                              .make_eqregion(origin,
                                             *representative,
                                             *region);
                }
            }

            // Replace the placeholder regions appearing in value with
            // their representatives
            let a_value =
                fold_regions_in(
                    self.tcx(),
                    &a_value,
                    |r, _| placeholder_representatives.get(&r).cloned().unwrap_or(r));

            debug!("higher_ranked_match: value={:?}", a_value);

            // We are now done with these placeholder variables.
            self.infcx.pop_placeholders(placeholder_map, snapshot);

            Ok(HrMatchResult { value: a_value })
        });
    }
}

fn fold_regions_in<'a, 'gcx, 'tcx, T, F>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                         unbound_value: &T,
                                         mut fldr: F)
                                         -> T
    where T: TypeFoldable<'tcx>,
          F: FnMut(ty::Region<'tcx>, ty::DebruijnIndex) -> ty::Region<'tcx>,
{
    tcx.fold_regions(unbound_value, &mut false, |region, current_depth| {
        // we should only be encountering "escaping" late-bound regions here,
        // because the ones at the current level should have been replaced
        // with fresh variables
        assert!(match *region {
            ty::ReLateBound(..) => false,
            _ => true
        });

        fldr(region, current_depth)
    })
}

impl<'a, 'gcx, 'tcx> InferCtxt<'a, 'gcx, 'tcx> {
    fn tainted_regions(&self,
                       snapshot: &CombinedSnapshot<'a, 'tcx>,
                       r: ty::Region<'tcx>,
                       directions: TaintDirections)
                       -> FxHashSet<ty::Region<'tcx>> {
        self.borrow_region_constraints().tainted(
            self.tcx,
            &snapshot.region_constraints_snapshot,
            r,
            directions)
    }

    fn region_vars_confined_to_snapshot(&self,
                                        snapshot: &CombinedSnapshot<'a, 'tcx>)
                                        -> Vec<ty::RegionVid>
    {
        /*!
         * Returns the set of region variables that do not affect any
         * types/regions which existed before `snapshot` was
         * started. This is used in the sub/lub/glb computations. The
         * idea here is that when we are computing lub/glb of two
         * regions, we sometimes create intermediate region variables.
         * Those region variables may touch some of the placeholder or
         * other "forbidden" regions we created to replace bound
         * regions, but they don't really represent an "external"
         * constraint.
         *
         * However, sometimes fresh variables are created for other
         * purposes too, and those *may* represent an external
         * constraint. In particular, when a type variable is
         * instantiated, we create region variables for all the
         * regions that appear within, and if that type variable
         * pre-existed the snapshot, then those region variables
         * represent external constraints.
         *
         * An example appears in the unit test
         * `sub_free_bound_false_infer`.  In this test, we want to
         * know whether
         *
         * ```rust
         * fn(_#0t) <: for<'a> fn(&'a int)
         * ```
         *
         * Note that the subtype has a type variable. Because the type
         * variable can't be instantiated with a region that is bound
         * in the fn signature, this comparison ought to fail. But if
         * we're not careful, it will succeed.
         *
         * The reason is that when we walk through the subtyping
         * algorithm, we begin by replacing `'a` with a placeholder
         * variable `'1`. We then have `fn(_#0t) <: fn(&'1 int)`. This
         * can be made true by unifying `_#0t` with `&'1 int`. In the
         * process, we create a fresh variable for the placeholder
         * region, `'$2`, and hence we have that `_#0t == &'$2
         * int`. However, because `'$2` was created during the sub
         * computation, if we're not careful we will erroneously
         * assume it is one of the transient region variables
         * representing a lub/glb internally. Not good.
         *
         * To prevent this, we check for type variables which were
         * unified during the snapshot, and say that any region
         * variable created during the snapshot but which finds its
         * way into a type variable is considered to "escape" the
         * snapshot.
         */

        let mut region_vars =
            self.borrow_region_constraints().vars_created_since_snapshot(
                &snapshot.region_constraints_snapshot);

        let escaping_types =
            self.type_variables.borrow_mut().types_escaping_snapshot(&snapshot.type_snapshot);

        let mut escaping_region_vars = FxHashSet::default();
        for ty in &escaping_types {
            self.tcx.collect_regions(ty, &mut escaping_region_vars);
        }

        region_vars.retain(|&region_vid| {
            let r = ty::ReVar(region_vid);
            !escaping_region_vars.contains(&r)
        });

        debug!("region_vars_confined_to_snapshot: region_vars={:?} escaping_types={:?}",
               region_vars,
               escaping_types);

        region_vars
    }

    /// Replace all regions (resp. types) bound by `binder` with placeholder
    /// regions (resp. types) and return a map indicating which bound-region
    /// was replaced with what placeholder region. This is the first step of
    /// checking subtyping when higher-ranked things are involved.
    ///
    /// **Important:** you must call this function from within a snapshot.
    /// Moreover, before committing the snapshot, you must eventually call
    /// either `plug_leaks` or `pop_placeholders` to remove the placeholder
    /// regions. If you rollback the snapshot (or are using a probe), then
    /// the pop occurs as part of the rollback, so an explicit call is not
    /// needed (but is also permitted).
    ///
    /// For more information about how placeholders and HRTBs work, see
    /// the [rustc guide].
    ///
    /// [rustc guide]: https://rust-lang.github.io/rustc-guide/traits/hrtb.html
    pub fn replace_bound_vars_with_placeholders<T>(
        &self,
        binder: &ty::Binder<T>
    ) -> (T, PlaceholderMap<'tcx>)
    where
        T: TypeFoldable<'tcx>
    {
        let next_universe = self.create_next_universe();

        let fld_r = |br| {
            self.tcx.mk_region(ty::RePlaceholder(ty::PlaceholderRegion {
                universe: next_universe,
                name: br,
            }))
        };

        let fld_t = |bound_ty: ty::BoundTy| {
            self.tcx.mk_ty(ty::Placeholder(ty::PlaceholderType {
                universe: next_universe,
                name: bound_ty.var,
            }))
        };

        let (result, map) = self.tcx.replace_bound_vars(binder, fld_r, fld_t);

        debug!(
            "replace_bound_vars_with_placeholders(binder={:?}, result={:?}, map={:?})",
            binder,
            result,
            map
        );

        (result, map)
    }

    /// Searches the region constraints created since `snapshot` was started
    /// and checks to determine whether any of the placeholder regions created
    /// in `placeholder_map` would "escape" -- meaning that they are related to
    /// other regions in some way. If so, the higher-ranked subtyping doesn't
    /// hold. See `README.md` for more details.
    pub fn leak_check(&self,
                      overly_polymorphic: bool,
                      _span: Span,
                      placeholder_map: &PlaceholderMap<'tcx>,
                      snapshot: &CombinedSnapshot<'a, 'tcx>)
                      -> RelateResult<'tcx, ()>
    {
        debug!("leak_check: placeholder_map={:?}",
               placeholder_map);

        // If the user gave `-Zno-leak-check`, then skip the leak
        // check completely. This is wildly unsound and also not
        // unlikely to cause an ICE or two. It is intended for use
        // only during a transition period, in which the MIR typeck
        // uses the "universe-style" check, and the rest of typeck
        // uses the more conservative leak check.  Since the leak
        // check is more conservative, we can't test the
        // universe-style check without disabling it.
        if self.tcx.sess.opts.debugging_opts.no_leak_check {
            return Ok(());
        }

        let new_vars = self.region_vars_confined_to_snapshot(snapshot);
        for (&placeholder_br, &placeholder) in placeholder_map {
            // The inputs to a placeholder variable can only
            // be itself or other new variables.
            let incoming_taints = self.tainted_regions(snapshot,
                                                       placeholder,
                                                       TaintDirections::both());
            for &tainted_region in &incoming_taints {
                // Each placeholder should only be relatable to itself
                // or new variables:
                match *tainted_region {
                    ty::ReVar(vid) => {
                        if new_vars.contains(&vid) {
                            continue;
                        }
                    }
                    _ => {
                        if tainted_region == placeholder { continue; }
                    }
                };

                debug!("{:?} (which replaced {:?}) is tainted by {:?}",
                       placeholder,
                       placeholder_br,
                       tainted_region);

                return Err(if overly_polymorphic {
                    debug!("Overly polymorphic!");
                    TypeError::RegionsOverlyPolymorphic(placeholder_br, tainted_region)
                } else {
                    debug!("Not as polymorphic!");
                    TypeError::RegionsInsufficientlyPolymorphic(placeholder_br, tainted_region)
                })
            }
        }

        Ok(())
    }

    /// This code converts from placeholder regions back to late-bound
    /// regions. It works by replacing each region in the taint set of a
    /// placeholder region with a bound-region. The bound region will be bound
    /// by the outer-most binder in `value`; the caller must ensure that there is
    /// such a binder and it is the right place.
    ///
    /// This routine is only intended to be used when the leak-check has
    /// passed; currently, it's used in the trait matching code to create
    /// a set of nested obligations from an impl that matches against
    /// something higher-ranked.  More details can be found in
    /// `librustc/middle/traits/README.md`.
    ///
    /// As a brief example, consider the obligation `for<'a> Fn(&'a int)
    /// -> &'a int`, and the impl:
    ///
    ///     impl<A,R> Fn<A,R> for SomethingOrOther
    ///         where A : Clone
    ///     { ... }
    ///
    /// Here we will have replaced `'a` with a placeholder region
    /// `'0`. This means that our substitution will be `{A=>&'0
    /// int, R=>&'0 int}`.
    ///
    /// When we apply the substitution to the bounds, we will wind up with
    /// `&'0 int : Clone` as a predicate. As a last step, we then go and
    /// replace `'0` with a late-bound region `'a`.  The depth is matched
    /// to the depth of the predicate, in this case 1, so that the final
    /// predicate is `for<'a> &'a int : Clone`.
    pub fn plug_leaks<T>(&self,
                         placeholder_map: PlaceholderMap<'tcx>,
                         snapshot: &CombinedSnapshot<'a, 'tcx>,
                         value: T) -> T
        where T : TypeFoldable<'tcx>
    {
        debug!("plug_leaks(placeholder_map={:?}, value={:?})",
               placeholder_map,
               value);

        if placeholder_map.is_empty() {
            return value;
        }

        // Compute a mapping from the "taint set" of each placeholder
        // region back to the `ty::BoundRegion` that it originally
        // represented. Because `leak_check` passed, we know that
        // these taint sets are mutually disjoint.
        let inv_placeholder_map: FxHashMap<ty::Region<'tcx>, ty::BoundRegion> =
            placeholder_map
            .iter()
            .flat_map(|(&placeholder_br, &placeholder)| {
                self.tainted_regions(snapshot, placeholder, TaintDirections::both())
                    .into_iter()
                    .map(move |tainted_region| (tainted_region, placeholder_br))
            })
            .collect();

        debug!("plug_leaks: inv_placeholder_map={:?}",
               inv_placeholder_map);

        // Remove any instantiated type variables from `value`; those can hide
        // references to regions from the `fold_regions` code below.
        let value = self.resolve_type_vars_if_possible(&value);

        // Map any placeholder byproducts back to a late-bound
        // region. Put that late-bound region at whatever the outermost
        // binder is that we encountered in `value`. The caller is
        // responsible for ensuring that (a) `value` contains at least one
        // binder and (b) that binder is the one we want to use.
        let result = self.tcx.fold_regions(&value, &mut false, |r, current_depth| {
            match inv_placeholder_map.get(&r) {
                None => r,
                Some(br) => {
                    // It is the responsibility of the caller to ensure
                    // that each placeholder region appears within a
                    // binder. In practice, this routine is only used by
                    // trait checking, and all of the placeholder regions
                    // appear inside predicates, which always have
                    // binders, so this assert is satisfied.
                    assert!(current_depth > ty::INNERMOST);

                    // since leak-check passed, this placeholder region
                    // should only have incoming edges from variables
                    // (which ought not to escape the snapshot, but we
                    // don't check that) or itself
                    assert!(
                        match *r {
                            ty::ReVar(_) => true,
                            ty::RePlaceholder(index) => index.name == *br,
                            _ => false,
                        },
                        "leak-check would have us replace {:?} with {:?}",
                        r, br);

                    self.tcx.mk_region(ty::ReLateBound(
                        current_depth.shifted_out(1),
                        br.clone(),
                    ))
                }
            }
        });

        self.pop_placeholders(placeholder_map, snapshot);

        debug!("plug_leaks: result={:?}", result);

        result
    }

    /// Pops the placeholder regions found in `placeholder_map` from the region
    /// inference context. Whenever you create placeholder regions via
    /// `replace_bound_vars_with_placeholders`, they must be popped before you
    /// commit the enclosing snapshot (if you do not commit, e.g., within a
    /// probe or as a result of an error, then this is not necessary, as
    /// popping happens as part of the rollback).
    ///
    /// Note: popping also occurs implicitly as part of `leak_check`.
    pub fn pop_placeholders(
        &self,
        placeholder_map: PlaceholderMap<'tcx>,
        snapshot: &CombinedSnapshot<'a, 'tcx>,
    ) {
        debug!("pop_placeholders({:?})", placeholder_map);
        let placeholder_regions: FxHashSet<_> = placeholder_map.values().cloned().collect();
        self.borrow_region_constraints().pop_placeholders(&placeholder_regions);
        self.universe.set(snapshot.universe);
        if !placeholder_map.is_empty() {
            self.projection_cache.borrow_mut().rollback_placeholder(
                &snapshot.projection_cache_snapshot);
        }
    }
}
