use crate::infer::outlives::free_region_map::FreeRegionMap;
use crate::infer::{GenericKind, InferCtxt};
use crate::hir;
use rustc_data_structures::fx::FxHashMap;
use syntax_pos::Span;
use crate::traits::query::outlives_bounds::{self, OutlivesBound};
use crate::ty::{self, Ty};

/// The `OutlivesEnvironment` collects information about what outlives
/// what in a given type-checking setting. For example, if we have a
/// where-clause like `where T: 'a` in scope, then the
/// `OutlivesEnvironment` would record that (in its
/// `region_bound_pairs` field). Similarly, it contains methods for
/// processing and adding implied bounds into the outlives
/// environment.
///
/// Other code at present does not typically take a
/// `&OutlivesEnvironment`, but rather takes some of its fields (e.g.,
/// `process_registered_region_obligations` wants the
/// region-bound-pairs). There is no mistaking it: the current setup
/// of tracking region information is quite scattered! The
/// `OutlivesEnvironment`, for example, needs to sometimes be combined
/// with the `middle::RegionRelations`, to yield a full picture of how
/// (lexical) lifetimes interact. However, I'm reluctant to do more
/// refactoring here, since the setup with NLL is quite different.
/// For example, NLL has no need of `RegionRelations`, and is solely
/// interested in the `OutlivesEnvironment`. -nmatsakis
#[derive(Clone)]
pub struct OutlivesEnvironment<'tcx> {
    param_env: ty::ParamEnv<'tcx>,
    free_region_map: FreeRegionMap<'tcx>,

    // Contains, for each body B that we are checking (that is, the fn
    // item, but also any nested closures), the set of implied region
    // bounds that are in scope in that particular body.
    //
    // Example:
    //
    // ```
    // fn foo<'a, 'b, T>(x: &'a T, y: &'b ()) {
    //   bar(x, y, |y: &'b T| { .. } // body B1)
    // } // body B0
    // ```
    //
    // Here, for body B0, the list would be `[T: 'a]`, because we
    // infer that `T` must outlive `'a` from the implied bounds on the
    // fn declaration.
    //
    // For the body B1, the list would be `[T: 'a, T: 'b]`, because we
    // also can see that -- within the closure body! -- `T` must
    // outlive `'b`. This is not necessarily true outside the closure
    // body, since the closure may never be called.
    //
    // We collect this map as we descend the tree. We then use the
    // results when proving outlives obligations like `T: 'x` later
    // (e.g., if `T: 'x` must be proven within the body B1, then we
    // know it is true if either `'a: 'x` or `'b: 'x`).
    region_bound_pairs_map: FxHashMap<hir::HirId, RegionBoundPairs<'tcx>>,

    // Used to compute `region_bound_pairs_map`: contains the set of
    // in-scope region-bound pairs thus far.
    region_bound_pairs_accum: RegionBoundPairs<'tcx>,
}

/// "Region-bound pairs" tracks outlives relations that are known to
/// be true, either because of explicit where-clauses like `T: 'a` or
/// because of implied bounds.
pub type RegionBoundPairs<'tcx> = Vec<(ty::Region<'tcx>, GenericKind<'tcx>)>;

impl<'a, 'tcx> OutlivesEnvironment<'tcx> {
    pub fn new(param_env: ty::ParamEnv<'tcx>) -> Self {
        let mut env = OutlivesEnvironment {
            param_env,
            free_region_map: Default::default(),
            region_bound_pairs_map: Default::default(),
            region_bound_pairs_accum: vec![],
        };

        env.add_outlives_bounds(None, outlives_bounds::explicit_outlives_bounds(param_env));

        env
    }

    /// Borrows current value of the `free_region_map`.
    pub fn free_region_map(&self) -> &FreeRegionMap<'tcx> {
        &self.free_region_map
    }

    /// Borrows current value of the `region_bound_pairs`.
    pub fn region_bound_pairs_map(&self) -> &FxHashMap<hir::HirId, RegionBoundPairs<'tcx>> {
        &self.region_bound_pairs_map
    }

    /// Returns ownership of the `free_region_map`.
    pub fn into_free_region_map(self) -> FreeRegionMap<'tcx> {
        self.free_region_map
    }

    /// This is a hack to support the old-skool regionck, which
    /// processes region constraints from the main function and the
    /// closure together. In that context, when we enter a closure, we
    /// want to be able to "save" the state of the surrounding a
    /// function. We can then add implied bounds and the like from the
    /// closure arguments into the environment -- these should only
    /// apply in the closure body, so once we exit, we invoke
    /// `pop_snapshot_post_closure` to remove them.
    ///
    /// Example:
    ///
    /// ```
    /// fn foo<T>() {
    ///    callback(for<'a> |x: &'a T| {
    ///         // ^^^^^^^ not legal syntax, but probably should be
    ///         // within this closure body, `T: 'a` holds
    ///    })
    /// }
    /// ```
    ///
    /// This "containment" of closure's effects only works so well. In
    /// particular, we (intentionally) leak relationships between free
    /// regions that are created by the closure's bounds. The case
    /// where this is useful is when you have (e.g.) a closure with a
    /// signature like `for<'a, 'b> fn(x: &'a &'b u32)` -- in this
    /// case, we want to keep the relationship `'b: 'a` in the
    /// free-region-map, so that later if we have to take `LUB('b,
    /// 'a)` we can get the result `'b`.
    ///
    /// I have opted to keep **all modifications** to the
    /// free-region-map, however, and not just those that concern free
    /// variables bound in the closure. The latter seems more correct,
    /// but it is not the existing behavior, and I could not find a
    /// case where the existing behavior went wrong. In any case, it
    /// seems like it'd be readily fixed if we wanted. There are
    /// similar leaks around givens that seem equally suspicious, to
    /// be honest. --nmatsakis
    pub fn push_snapshot_pre_closure(&self) -> usize {
        self.region_bound_pairs_accum.len()
    }

    /// See `push_snapshot_pre_closure`.
    pub fn pop_snapshot_post_closure(&mut self, len: usize) {
        self.region_bound_pairs_accum.truncate(len);
    }

    /// This method adds "implied bounds" into the outlives environment.
    /// Implied bounds are outlives relationships that we can deduce
    /// on the basis that certain types must be well-formed -- these are
    /// either the types that appear in the function signature or else
    /// the input types to an impl. For example, if you have a function
    /// like
    ///
    /// ```
    /// fn foo<'a, 'b, T>(x: &'a &'b [T]) { }
    /// ```
    ///
    /// we can assume in the caller's body that `'b: 'a` and that `T:
    /// 'b` (and hence, transitively, that `T: 'a`). This method would
    /// add those assumptions into the outlives-environment.
    ///
    /// Tests: `src/test/compile-fail/regions-free-region-ordering-*.rs`
    pub fn add_implied_bounds(
        &mut self,
        infcx: &InferCtxt<'a, 'tcx>,
        fn_sig_tys: &[Ty<'tcx>],
        body_id: hir::HirId,
        span: Span,
    ) {
        debug!("add_implied_bounds()");

        for &ty in fn_sig_tys {
            let ty = infcx.resolve_vars_if_possible(&ty);
            debug!("add_implied_bounds: ty = {}", ty);
            let implied_bounds = infcx.implied_outlives_bounds(self.param_env, body_id, ty, span);
            self.add_outlives_bounds(Some(infcx), implied_bounds)
        }
    }

    /// Save the current set of region-bound pairs under the given `body_id`.
    pub fn save_implied_bounds(&mut self, body_id: hir::HirId) {
        let old = self.region_bound_pairs_map.insert(
            body_id,
            self.region_bound_pairs_accum.clone(),
        );
        assert!(old.is_none());
    }

    /// Processes outlives bounds that are known to hold, whether from implied or other sources.
    ///
    /// The `infcx` parameter is optional; if the implied bounds may
    /// contain inference variables, it must be supplied, in which
    /// case we will register "givens" on the inference context. (See
    /// `RegionConstraintData`.)
    fn add_outlives_bounds<I>(&mut self, infcx: Option<&InferCtxt<'a, 'tcx>>, outlives_bounds: I)
    where
        I: IntoIterator<Item = OutlivesBound<'tcx>>,
    {
        // Record relationships such as `T:'x` that don't go into the
        // free-region-map but which we use here.
        for outlives_bound in outlives_bounds {
            debug!("add_outlives_bounds: outlives_bound={:?}", outlives_bound);
            match outlives_bound {
                OutlivesBound::RegionSubRegion(r_a @ &ty::ReEarlyBound(_), &ty::ReVar(vid_b))
                | OutlivesBound::RegionSubRegion(r_a @ &ty::ReFree(_), &ty::ReVar(vid_b)) => {
                    infcx
                        .expect("no infcx provided but region vars found")
                        .add_given(r_a, vid_b);
                }
                OutlivesBound::RegionSubParam(r_a, param_b) => {
                    self.region_bound_pairs_accum
                        .push((r_a, GenericKind::Param(param_b)));
                }
                OutlivesBound::RegionSubProjection(r_a, projection_b) => {
                    self.region_bound_pairs_accum
                        .push((r_a, GenericKind::Projection(projection_b)));
                }
                OutlivesBound::RegionSubRegion(r_a, r_b) => {
                    // In principle, we could record (and take
                    // advantage of) every relationship here, but
                    // we are also free not to -- it simply means
                    // strictly less that we can successfully type
                    // check. Right now we only look for things
                    // relationships between free regions. (It may
                    // also be that we should revise our inference
                    // system to be more general and to make use
                    // of *every* relationship that arises here,
                    // but presently we do not.)
                    self.free_region_map.relate_regions(r_a, r_b);
                }
            }
        }
    }
}
