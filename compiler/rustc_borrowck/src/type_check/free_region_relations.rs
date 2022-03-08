use rustc_data_structures::frozen::Frozen;
use rustc_data_structures::transitive_relation::TransitiveRelation;
use rustc_infer::infer::canonical::QueryRegionConstraints;
use rustc_infer::infer::outlives;
use rustc_infer::infer::region_constraints::GenericKind;
use rustc_infer::infer::InferCtxt;
use rustc_middle::mir::ConstraintCategory;
use rustc_middle::traits::query::OutlivesBound;
use rustc_middle::ty::{self, RegionVid, Ty};
use rustc_span::DUMMY_SP;
use rustc_trait_selection::traits::query::type_op::{self, TypeOp};
use std::rc::Rc;
use type_op::TypeOpOutput;

use crate::{
    type_check::constraint_conversion,
    type_check::{Locations, MirTypeckRegionConstraints},
    universal_regions::UniversalRegions,
};

#[derive(Debug)]
crate struct UniversalRegionRelations<'tcx> {
    universal_regions: Rc<UniversalRegions<'tcx>>,

    /// Stores the outlives relations that are known to hold from the
    /// implied bounds, in-scope where-clauses, and that sort of
    /// thing.
    outlives: TransitiveRelation<RegionVid>,

    /// This is the `<=` relation; that is, if `a: b`, then `b <= a`,
    /// and we store that here. This is useful when figuring out how
    /// to express some local region in terms of external regions our
    /// caller will understand.
    inverse_outlives: TransitiveRelation<RegionVid>,
}

/// Each RBP `('a, GK)` indicates that `GK: 'a` can be assumed to
/// be true. These encode relationships like `T: 'a` that are
/// added via implicit bounds.
///
/// Each region here is guaranteed to be a key in the `indices`
/// map. We use the "original" regions (i.e., the keys from the
/// map, and not the values) because the code in
/// `process_registered_region_obligations` has some special-cased
/// logic expecting to see (e.g.) `ReStatic`, and if we supplied
/// our special inference variable there, we would mess that up.
type RegionBoundPairs<'tcx> = Vec<(ty::Region<'tcx>, GenericKind<'tcx>)>;

/// As part of computing the free region relations, we also have to
/// normalize the input-output types, which we then need later. So we
/// return those. This vector consists of first the input types and
/// then the output type as the last element.
type NormalizedInputsAndOutput<'tcx> = Vec<Ty<'tcx>>;

crate struct CreateResult<'tcx> {
    crate universal_region_relations: Frozen<UniversalRegionRelations<'tcx>>,
    crate region_bound_pairs: RegionBoundPairs<'tcx>,
    crate normalized_inputs_and_output: NormalizedInputsAndOutput<'tcx>,
}

crate fn create<'tcx>(
    infcx: &InferCtxt<'_, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    implicit_region_bound: Option<ty::Region<'tcx>>,
    universal_regions: &Rc<UniversalRegions<'tcx>>,
    constraints: &mut MirTypeckRegionConstraints<'tcx>,
) -> CreateResult<'tcx> {
    UniversalRegionRelationsBuilder {
        infcx,
        param_env,
        implicit_region_bound,
        constraints,
        universal_regions: universal_regions.clone(),
        region_bound_pairs: Vec::new(),
        relations: UniversalRegionRelations {
            universal_regions: universal_regions.clone(),
            outlives: Default::default(),
            inverse_outlives: Default::default(),
        },
    }
    .create()
}

impl UniversalRegionRelations<'_> {
    /// Records in the `outlives_relation` (and
    /// `inverse_outlives_relation`) that `fr_a: fr_b`. Invoked by the
    /// builder below.
    fn relate_universal_regions(&mut self, fr_a: RegionVid, fr_b: RegionVid) {
        debug!("relate_universal_regions: fr_a={:?} outlives fr_b={:?}", fr_a, fr_b);
        self.outlives.add(fr_a, fr_b);
        self.inverse_outlives.add(fr_b, fr_a);
    }

    /// Given two universal regions, returns the postdominating
    /// upper-bound (effectively the least upper bound).
    ///
    /// (See `TransitiveRelation::postdom_upper_bound` for details on
    /// the postdominating upper bound in general.)
    crate fn postdom_upper_bound(&self, fr1: RegionVid, fr2: RegionVid) -> RegionVid {
        assert!(self.universal_regions.is_universal_region(fr1));
        assert!(self.universal_regions.is_universal_region(fr2));
        *self
            .inverse_outlives
            .postdom_upper_bound(&fr1, &fr2)
            .unwrap_or(&self.universal_regions.fr_static)
    }

    /// Finds an "upper bound" for `fr` that is not local. In other
    /// words, returns the smallest (*) known region `fr1` that (a)
    /// outlives `fr` and (b) is not local.
    ///
    /// (*) If there are multiple competing choices, we return all of them.
    crate fn non_local_upper_bounds<'a>(&'a self, fr: &'a RegionVid) -> Vec<&'a RegionVid> {
        debug!("non_local_upper_bound(fr={:?})", fr);
        let res = self.non_local_bounds(&self.inverse_outlives, fr);
        assert!(!res.is_empty(), "can't find an upper bound!?");
        res
    }

    /// Returns the "postdominating" bound of the set of
    /// `non_local_upper_bounds` for the given region.
    crate fn non_local_upper_bound(&self, fr: RegionVid) -> RegionVid {
        let upper_bounds = self.non_local_upper_bounds(&fr);

        // In case we find more than one, reduce to one for
        // convenience.  This is to prevent us from generating more
        // complex constraints, but it will cause spurious errors.
        let post_dom = self.inverse_outlives.mutual_immediate_postdominator(upper_bounds);

        debug!("non_local_bound: post_dom={:?}", post_dom);

        post_dom
            .and_then(|&post_dom| {
                // If the mutual immediate postdom is not local, then
                // there is no non-local result we can return.
                if !self.universal_regions.is_local_free_region(post_dom) {
                    Some(post_dom)
                } else {
                    None
                }
            })
            .unwrap_or(self.universal_regions.fr_static)
    }

    /// Finds a "lower bound" for `fr` that is not local. In other
    /// words, returns the largest (*) known region `fr1` that (a) is
    /// outlived by `fr` and (b) is not local.
    ///
    /// (*) If there are multiple competing choices, we pick the "postdominating"
    /// one. See `TransitiveRelation::postdom_upper_bound` for details.
    crate fn non_local_lower_bound(&self, fr: RegionVid) -> Option<RegionVid> {
        debug!("non_local_lower_bound(fr={:?})", fr);
        let lower_bounds = self.non_local_bounds(&self.outlives, &fr);

        // In case we find more than one, reduce to one for
        // convenience.  This is to prevent us from generating more
        // complex constraints, but it will cause spurious errors.
        let post_dom = self.outlives.mutual_immediate_postdominator(lower_bounds);

        debug!("non_local_bound: post_dom={:?}", post_dom);

        post_dom.and_then(|&post_dom| {
            // If the mutual immediate postdom is not local, then
            // there is no non-local result we can return.
            if !self.universal_regions.is_local_free_region(post_dom) {
                Some(post_dom)
            } else {
                None
            }
        })
    }

    /// Helper for `non_local_upper_bounds` and `non_local_lower_bounds`.
    /// Repeatedly invokes `postdom_parent` until we find something that is not
    /// local. Returns `None` if we never do so.
    fn non_local_bounds<'a>(
        &self,
        relation: &'a TransitiveRelation<RegionVid>,
        fr0: &'a RegionVid,
    ) -> Vec<&'a RegionVid> {
        // This method assumes that `fr0` is one of the universally
        // quantified region variables.
        assert!(self.universal_regions.is_universal_region(*fr0));

        let mut external_parents = vec![];
        let mut queue = vec![fr0];

        // Keep expanding `fr` into its parents until we reach
        // non-local regions.
        while let Some(fr) = queue.pop() {
            if !self.universal_regions.is_local_free_region(*fr) {
                external_parents.push(fr);
                continue;
            }

            queue.extend(relation.parents(fr));
        }

        debug!("non_local_bound: external_parents={:?}", external_parents);

        external_parents
    }

    /// Returns `true` if fr1 is known to outlive fr2.
    ///
    /// This will only ever be true for universally quantified regions.
    crate fn outlives(&self, fr1: RegionVid, fr2: RegionVid) -> bool {
        self.outlives.contains(&fr1, &fr2)
    }

    /// Returns a vector of free regions `x` such that `fr1: x` is
    /// known to hold.
    crate fn regions_outlived_by(&self, fr1: RegionVid) -> Vec<&RegionVid> {
        self.outlives.reachable_from(&fr1)
    }

    /// Returns the _non-transitive_ set of known `outlives` constraints between free regions.
    crate fn known_outlives(&self) -> impl Iterator<Item = (&RegionVid, &RegionVid)> {
        self.outlives.base_edges()
    }
}

struct UniversalRegionRelationsBuilder<'this, 'tcx> {
    infcx: &'this InferCtxt<'this, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    universal_regions: Rc<UniversalRegions<'tcx>>,
    implicit_region_bound: Option<ty::Region<'tcx>>,
    constraints: &'this mut MirTypeckRegionConstraints<'tcx>,

    // outputs:
    relations: UniversalRegionRelations<'tcx>,
    region_bound_pairs: RegionBoundPairs<'tcx>,
}

impl<'tcx> UniversalRegionRelationsBuilder<'_, 'tcx> {
    crate fn create(mut self) -> CreateResult<'tcx> {
        let unnormalized_input_output_tys = self
            .universal_regions
            .unnormalized_input_tys
            .iter()
            .cloned()
            .chain(Some(self.universal_regions.unnormalized_output_ty));

        // For each of the input/output types:
        // - Normalize the type. This will create some region
        //   constraints, which we buffer up because we are
        //   not ready to process them yet.
        // - Then compute the implied bounds. This will adjust
        //   the `region_bound_pairs` and so forth.
        // - After this is done, we'll process the constraints, once
        //   the `relations` is built.
        let mut normalized_inputs_and_output =
            Vec::with_capacity(self.universal_regions.unnormalized_input_tys.len() + 1);
        let constraint_sets: Vec<_> = unnormalized_input_output_tys
            .flat_map(|ty| {
                debug!("build: input_or_output={:?}", ty);
                // We add implied bounds from both the unnormalized and normalized ty
                // See issue #87748
                let TypeOpOutput { output: norm_ty, constraints: constraints1, .. } = self
                    .param_env
                    .and(type_op::normalize::Normalize::new(ty))
                    .fully_perform(self.infcx)
                    .unwrap_or_else(|_| {
                        self.infcx
                            .tcx
                            .sess
                            .delay_span_bug(DUMMY_SP, &format!("failed to normalize {:?}", ty));
                        TypeOpOutput {
                            output: self.infcx.tcx.ty_error(),
                            constraints: None,
                            canonicalized_query: None,
                        }
                    });
                // Note: we need this in examples like
                // ```
                // trait Foo {
                //   type Bar;
                //   fn foo(&self) -> &Self::Bar;
                // }
                // impl Foo for () {
                //   type Bar = ();
                //   fn foo(&self) ->&() {}
                // }
                // ```
                // Both &Self::Bar and &() are WF
                let constraints_implied = self.add_implied_bounds(norm_ty);
                normalized_inputs_and_output.push(norm_ty);
                constraints1.into_iter().chain(constraints_implied)
            })
            .collect();

        // Insert the facts we know from the predicates. Why? Why not.
        let param_env = self.param_env;
        self.add_outlives_bounds(outlives::explicit_outlives_bounds(param_env));

        // Finally:
        // - outlives is reflexive, so `'r: 'r` for every region `'r`
        // - `'static: 'r` for every region `'r`
        // - `'r: 'fn_body` for every (other) universally quantified
        //   region `'r`, all of which are provided by our caller
        let fr_static = self.universal_regions.fr_static;
        let fr_fn_body = self.universal_regions.fr_fn_body;
        for fr in self.universal_regions.universal_regions() {
            debug!("build: relating free region {:?} to itself and to 'static", fr);
            self.relations.relate_universal_regions(fr, fr);
            self.relations.relate_universal_regions(fr_static, fr);
            self.relations.relate_universal_regions(fr, fr_fn_body);
        }

        for data in &constraint_sets {
            constraint_conversion::ConstraintConversion::new(
                self.infcx,
                &self.universal_regions,
                &self.region_bound_pairs,
                self.implicit_region_bound,
                self.param_env,
                Locations::All(DUMMY_SP),
                ConstraintCategory::Internal,
                &mut self.constraints,
            )
            .convert_all(data);
        }

        CreateResult {
            universal_region_relations: Frozen::freeze(self.relations),
            region_bound_pairs: self.region_bound_pairs,
            normalized_inputs_and_output,
        }
    }

    /// Update the type of a single local, which should represent
    /// either the return type of the MIR or one of its arguments. At
    /// the same time, compute and add any implied bounds that come
    /// from this local.
    fn add_implied_bounds(&mut self, ty: Ty<'tcx>) -> Option<Rc<QueryRegionConstraints<'tcx>>> {
        debug!("add_implied_bounds(ty={:?})", ty);
        let TypeOpOutput { output: bounds, constraints, .. } = self
            .param_env
            .and(type_op::implied_outlives_bounds::ImpliedOutlivesBounds { ty })
            .fully_perform(self.infcx)
            .unwrap_or_else(|_| bug!("failed to compute implied bounds {:?}", ty));
        self.add_outlives_bounds(bounds);
        constraints
    }

    /// Registers the `OutlivesBound` items from `outlives_bounds` in
    /// the outlives relation as well as the region-bound pairs
    /// listing.
    fn add_outlives_bounds<I>(&mut self, outlives_bounds: I)
    where
        I: IntoIterator<Item = OutlivesBound<'tcx>>,
    {
        for outlives_bound in outlives_bounds {
            debug!("add_outlives_bounds(bound={:?})", outlives_bound);

            match outlives_bound {
                OutlivesBound::RegionSubRegion(r1, r2) => {
                    // `where Type:` is lowered to `where Type: 'empty` so that
                    // we check `Type` is well formed, but there's no use for
                    // this bound here.
                    if r1.is_empty() {
                        return;
                    }

                    // The bound says that `r1 <= r2`; we store `r2: r1`.
                    let r1 = self.universal_regions.to_region_vid(r1);
                    let r2 = self.universal_regions.to_region_vid(r2);
                    self.relations.relate_universal_regions(r2, r1);
                }

                OutlivesBound::RegionSubParam(r_a, param_b) => {
                    self.region_bound_pairs.push((r_a, GenericKind::Param(param_b)));
                }

                OutlivesBound::RegionSubProjection(r_a, projection_b) => {
                    self.region_bound_pairs.push((r_a, GenericKind::Projection(projection_b)));
                }
            }
        }
    }
}
