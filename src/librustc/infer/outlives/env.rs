// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use infer::{GenericKind, InferCtxt};
use infer::outlives::free_region_map::FreeRegionMap;
use infer::outlives::bounds::{self, OutlivesBound};
use ty::{self, Ty};

use syntax::ast;
use syntax_pos::Span;

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
    region_bound_pairs: Vec<(ty::Region<'tcx>, GenericKind<'tcx>)>,
}

impl<'a, 'gcx: 'tcx, 'tcx: 'a> OutlivesEnvironment<'tcx> {
    pub fn new(param_env: ty::ParamEnv<'tcx>) -> Self {
        let mut env = OutlivesEnvironment {
            param_env,
            free_region_map: FreeRegionMap::new(),
            region_bound_pairs: vec![],
        };

        env.add_outlives_bounds(None, bounds::explicit_outlives_bounds(param_env));

        env
    }

    /// Borrows current value of the `free_region_map`.
    pub fn free_region_map(&self) -> &FreeRegionMap<'tcx> {
        &self.free_region_map
    }

    /// Borrows current value of the `region_bound_pairs`.
    pub fn region_bound_pairs(&self) -> &[(ty::Region<'tcx>, GenericKind<'tcx>)] {
        &self.region_bound_pairs
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
        self.region_bound_pairs.len()
    }

    /// See `push_snapshot_pre_closure`.
    pub fn pop_snapshot_post_closure(&mut self, len: usize) {
        self.region_bound_pairs.truncate(len);
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
        infcx: &InferCtxt<'a, 'gcx, 'tcx>,
        fn_sig_tys: &[Ty<'tcx>],
        body_id: ast::NodeId,
        span: Span,
    ) {
        debug!("add_implied_bounds()");

        for &ty in fn_sig_tys {
            let ty = infcx.resolve_type_vars_if_possible(&ty);
            debug!("add_implied_bounds: ty = {}", ty);
            let implied_bounds = infcx.implied_outlives_bounds(self.param_env, body_id, ty, span);
            self.add_outlives_bounds(Some(infcx), implied_bounds)
        }
    }

    /// Processes outlives bounds that are known to hold, whether from implied or other sources.
    ///
    /// The `infcx` parameter is optional; if the implied bounds may
    /// contain inference variables, it must be supplied, in which
    /// case we will register "givens" on the inference context. (See
    /// `RegionConstraintData`.)
    fn add_outlives_bounds<I>(
        &mut self,
        infcx: Option<&InferCtxt<'a, 'gcx, 'tcx>>,
        outlives_bounds: I,
    ) where
        I: IntoIterator<Item = OutlivesBound<'tcx>>,
    {
        // Record relationships such as `T:'x` that don't go into the
        // free-region-map but which we use here.
        for outlives_bound in outlives_bounds {
            debug!("add_outlives_bounds: outlives_bound={:?}", outlives_bound);
            match outlives_bound {
                OutlivesBound::RegionSubRegion(r_a @ &ty::ReEarlyBound(_), &ty::ReVar(vid_b)) |
                OutlivesBound::RegionSubRegion(r_a @ &ty::ReFree(_), &ty::ReVar(vid_b)) => {
                    infcx.expect("no infcx provided but region vars found").add_given(r_a, vid_b);
                }
                OutlivesBound::RegionSubParam(r_a, param_b) => {
                    self.region_bound_pairs
                        .push((r_a, GenericKind::Param(param_b)));
                }
                OutlivesBound::RegionSubProjection(r_a, projection_b) => {
                    self.region_bound_pairs
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
