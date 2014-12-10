// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Helper routines for higher-ranked things. See the `doc` module at
//! the end of the file for details.

use super::{combine, CombinedSnapshot, cres, InferCtxt, HigherRankedType};
use super::combine::Combine;

use middle::ty::{mod, Ty, replace_late_bound_regions};
use middle::ty_fold::{mod, HigherRankedFoldable, TypeFoldable};
use syntax::codemap::Span;
use util::nodemap::{FnvHashMap, FnvHashSet};
use util::ppaux::{bound_region_to_string, Repr};

pub trait HigherRankedCombineable<'tcx>: HigherRankedFoldable<'tcx> +
                                         TypeFoldable<'tcx> + Repr<'tcx> {
    fn super_combine<C:Combine<'tcx>>(combiner: &C, a: &Self, b: &Self) -> cres<'tcx, Self>;
}

pub trait HigherRankedRelations<'tcx> {
    fn higher_ranked_sub<T>(&self, a: &T, b: &T) -> cres<'tcx, T>
        where T : HigherRankedCombineable<'tcx>;

    fn higher_ranked_lub<T>(&self, a: &T, b: &T) -> cres<'tcx, T>
        where T : HigherRankedCombineable<'tcx>;

    fn higher_ranked_glb<T>(&self, a: &T, b: &T) -> cres<'tcx, T>
        where T : HigherRankedCombineable<'tcx>;
}

trait InferCtxtExt<'tcx> {
    fn tainted_regions(&self, snapshot: &CombinedSnapshot, r: ty::Region) -> Vec<ty::Region>;

    fn region_vars_confined_to_snapshot(&self,
                                        snapshot: &CombinedSnapshot)
                                        -> Vec<ty::RegionVid>;
}

impl<'tcx,C> HigherRankedRelations<'tcx> for C
    where C : Combine<'tcx>
{
    fn higher_ranked_sub<T>(&self, a: &T, b: &T) -> cres<'tcx, T>
        where T : HigherRankedCombineable<'tcx>
    {
        debug!("higher_ranked_sub(a={}, b={})",
               a.repr(self.tcx()), b.repr(self.tcx()));

        // Rather than checking the subtype relationship between `a` and `b`
        // as-is, we need to do some extra work here in order to make sure
        // that function subtyping works correctly with respect to regions
        //
        // Note: this is a subtle algorithm.  For a full explanation,
        // please see the large comment at the end of the file in the (inlined) module
        // `doc`.

        // Start a snapshot so we can examine "all bindings that were
        // created as part of this type comparison".
        return self.infcx().try(|snapshot| {
            // First, we instantiate each bound region in the subtype with a fresh
            // region variable.
            let (a_prime, _) =
                self.infcx().replace_late_bound_regions_with_fresh_var(
                    self.trace().origin.span(),
                    HigherRankedType,
                    a);

            // Second, we instantiate each bound region in the supertype with a
            // fresh concrete region.
            let (b_prime, skol_map) = {
                replace_late_bound_regions(self.tcx(), b, |br, _| {
                    let skol =
                        self.infcx().region_vars.new_skolemized(
                            br, &snapshot.region_vars_snapshot);
                    debug!("Bound region {} skolemized to {}",
                           bound_region_to_string(self.tcx(), "", false, br),
                           skol);
                    skol
                })
            };

            debug!("a_prime={}", a_prime.repr(self.tcx()));
            debug!("b_prime={}", b_prime.repr(self.tcx()));

            // Compare types now that bound regions have been replaced.
            let result = try!(HigherRankedCombineable::super_combine(self, &a_prime, &b_prime));

            // Presuming type comparison succeeds, we need to check
            // that the skolemized regions do not "leak".
            let new_vars = self.infcx().region_vars_confined_to_snapshot(snapshot);
            for (&skol_br, &skol) in skol_map.iter() {
                let tainted = self.infcx().tainted_regions(snapshot, skol);
                for tainted_region in tainted.iter() {
                    // Each skolemized should only be relatable to itself
                    // or new variables:
                    match *tainted_region {
                        ty::ReInfer(ty::ReVar(ref vid)) => {
                            if new_vars.iter().any(|x| x == vid) { continue; }
                        }
                        _ => {
                            if *tainted_region == skol { continue; }
                        }
                    };

                    // A is not as polymorphic as B:
                    if self.a_is_expected() {
                        debug!("Not as polymorphic!");
                        return Err(ty::terr_regions_insufficiently_polymorphic(skol_br,
                                                                               *tainted_region));
                    } else {
                        debug!("Overly polymorphic!");
                        return Err(ty::terr_regions_overly_polymorphic(skol_br,
                                                                       *tainted_region));
                    }
                }
            }

            debug!("higher_ranked_sub: OK result={}",
                   result.repr(self.tcx()));

            Ok(result)
        });
    }

    fn higher_ranked_lub<T>(&self, a: &T, b: &T) -> cres<'tcx, T>
        where T : HigherRankedCombineable<'tcx>
    {
        // Start a snapshot so we can examine "all bindings that were
        // created as part of this type comparison".
        return self.infcx().try(|snapshot| {
            // Instantiate each bound region with a fresh region variable.
            let span = self.trace().origin.span();
            let (a_with_fresh, a_map) =
                self.infcx().replace_late_bound_regions_with_fresh_var(
                    span, HigherRankedType, a);
            let (b_with_fresh, _) =
                self.infcx().replace_late_bound_regions_with_fresh_var(
                    span, HigherRankedType, b);

            // Collect constraints.
            let result0 =
                try!(HigherRankedCombineable::super_combine(self, &a_with_fresh, &b_with_fresh));
            let result0 =
                self.infcx().resolve_type_vars_if_possible(&result0);
            debug!("lub result0 = {}", result0.repr(self.tcx()));

            // Generalize the regions appearing in result0 if possible
            let new_vars = self.infcx().region_vars_confined_to_snapshot(snapshot);
            let span = self.trace().origin.span();
            let result1 =
                fold_regions_in(
                    self.tcx(),
                    &result0,
                    |r, debruijn| generalize_region(self.infcx(), span, snapshot, debruijn,
                                                    new_vars.as_slice(), &a_map, r));

            debug!("lub({},{}) = {}",
                   a.repr(self.tcx()),
                   b.repr(self.tcx()),
                   result1.repr(self.tcx()));

            Ok(result1)
        });

        fn generalize_region(infcx: &InferCtxt,
                             span: Span,
                             snapshot: &CombinedSnapshot,
                             debruijn: ty::DebruijnIndex,
                             new_vars: &[ty::RegionVid],
                             a_map: &FnvHashMap<ty::BoundRegion, ty::Region>,
                             r0: ty::Region)
                             -> ty::Region {
            // Regions that pre-dated the LUB computation stay as they are.
            if !is_var_in_set(new_vars, r0) {
                assert!(!r0.is_bound());
                debug!("generalize_region(r0={}): not new variable", r0);
                return r0;
            }

            let tainted = infcx.tainted_regions(snapshot, r0);

            // Variables created during LUB computation which are
            // *related* to regions that pre-date the LUB computation
            // stay as they are.
            if !tainted.iter().all(|r| is_var_in_set(new_vars, *r)) {
                debug!("generalize_region(r0={}): \
                        non-new-variables found in {}",
                       r0, tainted);
                assert!(!r0.is_bound());
                return r0;
            }

            // Otherwise, the variable must be associated with at
            // least one of the variables representing bound regions
            // in both A and B.  Replace the variable with the "first"
            // bound region from A that we find it to be associated
            // with.
            for (a_br, a_r) in a_map.iter() {
                if tainted.iter().any(|x| x == a_r) {
                    debug!("generalize_region(r0={}): \
                            replacing with {}, tainted={}",
                           r0, *a_br, tainted);
                    return ty::ReLateBound(debruijn, *a_br);
                }
            }

            infcx.tcx.sess.span_bug(
                span,
                format!("region {} is not associated with \
                         any bound region from A!",
                        r0).as_slice())
        }
    }

    fn higher_ranked_glb<T>(&self, a: &T, b: &T) -> cres<'tcx, T>
        where T : HigherRankedCombineable<'tcx>
    {
        debug!("{}.higher_ranked_glb({}, {})",
               self.tag(), a.repr(self.tcx()), b.repr(self.tcx()));

        // Make a snapshot so we can examine "all bindings that were
        // created as part of this type comparison".
        return self.infcx().try(|snapshot| {
            // Instantiate each bound region with a fresh region variable.
            let (a_with_fresh, a_map) =
                self.infcx().replace_late_bound_regions_with_fresh_var(
                    self.trace().origin.span(), HigherRankedType, a);
            let (b_with_fresh, b_map) =
                self.infcx().replace_late_bound_regions_with_fresh_var(
                    self.trace().origin.span(), HigherRankedType, b);
            let a_vars = var_ids(self, &a_map);
            let b_vars = var_ids(self, &b_map);

            // Collect constraints.
            let result0 =
                try!(HigherRankedCombineable::super_combine(self, &a_with_fresh, &b_with_fresh));
            let result0 =
                self.infcx().resolve_type_vars_if_possible(&result0);
            debug!("glb result0 = {}", result0.repr(self.tcx()));

            // Generalize the regions appearing in fn_ty0 if possible
            let new_vars = self.infcx().region_vars_confined_to_snapshot(snapshot);
            let span = self.trace().origin.span();
            let result1 =
                fold_regions_in(
                    self.tcx(),
                    &result0,
                    |r, debruijn| generalize_region(self.infcx(), span, snapshot, debruijn,
                                                    new_vars.as_slice(),
                                                    &a_map, a_vars.as_slice(), b_vars.as_slice(),
                                                    r));

            debug!("glb({},{}) = {}",
                   a.repr(self.tcx()),
                   b.repr(self.tcx()),
                   result1.repr(self.tcx()));

            Ok(result1)
        });

        fn generalize_region(infcx: &InferCtxt,
                             span: Span,
                             snapshot: &CombinedSnapshot,
                             debruijn: ty::DebruijnIndex,
                             new_vars: &[ty::RegionVid],
                             a_map: &FnvHashMap<ty::BoundRegion, ty::Region>,
                             a_vars: &[ty::RegionVid],
                             b_vars: &[ty::RegionVid],
                             r0: ty::Region) -> ty::Region {
            if !is_var_in_set(new_vars, r0) {
                assert!(!r0.is_bound());
                return r0;
            }

            let tainted = infcx.tainted_regions(snapshot, r0);

            let mut a_r = None;
            let mut b_r = None;
            let mut only_new_vars = true;
            for r in tainted.iter() {
                if is_var_in_set(a_vars, *r) {
                    if a_r.is_some() {
                        return fresh_bound_variable(infcx, debruijn);
                    } else {
                        a_r = Some(*r);
                    }
                } else if is_var_in_set(b_vars, *r) {
                    if b_r.is_some() {
                        return fresh_bound_variable(infcx, debruijn);
                    } else {
                        b_r = Some(*r);
                    }
                } else if !is_var_in_set(new_vars, *r) {
                    only_new_vars = false;
                }
            }

            // NB---I do not believe this algorithm computes
            // (necessarily) the GLB.  As written it can
            // spuriously fail. In particular, if there is a case
            // like: |fn(&a)| and fn(fn(&b)), where a and b are
            // free, it will return fn(&c) where c = GLB(a,b).  If
            // however this GLB is not defined, then the result is
            // an error, even though something like
            // "fn<X>(fn(&X))" where X is bound would be a
            // subtype of both of those.
            //
            // The problem is that if we were to return a bound
            // variable, we'd be computing a lower-bound, but not
            // necessarily the *greatest* lower-bound.
            //
            // Unfortunately, this problem is non-trivial to solve,
            // because we do not know at the time of computing the GLB
            // whether a GLB(a,b) exists or not, because we haven't
            // run region inference (or indeed, even fully computed
            // the region hierarchy!). The current algorithm seems to
            // works ok in practice.

            if a_r.is_some() && b_r.is_some() && only_new_vars {
                // Related to exactly one bound variable from each fn:
                return rev_lookup(infcx, span, a_map, a_r.unwrap());
            } else if a_r.is_none() && b_r.is_none() {
                // Not related to bound variables from either fn:
                assert!(!r0.is_bound());
                return r0;
            } else {
                // Other:
                return fresh_bound_variable(infcx, debruijn);
            }
        }

        fn rev_lookup(infcx: &InferCtxt,
                      span: Span,
                      a_map: &FnvHashMap<ty::BoundRegion, ty::Region>,
                      r: ty::Region) -> ty::Region
        {
            for (a_br, a_r) in a_map.iter() {
                if *a_r == r {
                    return ty::ReLateBound(ty::DebruijnIndex::new(1), *a_br);
                }
            }
            infcx.tcx.sess.span_bug(
                span,
                format!("could not find original bound region for {}", r)[]);
        }

        fn fresh_bound_variable(infcx: &InferCtxt, debruijn: ty::DebruijnIndex) -> ty::Region {
            infcx.region_vars.new_bound(debruijn)
        }
    }
}

impl<'tcx> HigherRankedCombineable<'tcx> for ty::FnSig<'tcx> {
    fn super_combine<C:Combine<'tcx>>(combiner: &C, a: &ty::FnSig<'tcx>, b: &ty::FnSig<'tcx>)
                                      -> cres<'tcx, ty::FnSig<'tcx>>
    {
        if a.variadic != b.variadic {
            return Err(ty::terr_variadic_mismatch(
                combine::expected_found(combiner, a.variadic, b.variadic)));
        }

        let inputs = try!(argvecs(combiner,
                                  a.inputs.as_slice(),
                                  b.inputs.as_slice()));

        let output = try!(match (a.output, b.output) {
            (ty::FnConverging(a_ty), ty::FnConverging(b_ty)) =>
                Ok(ty::FnConverging(try!(combiner.tys(a_ty, b_ty)))),
            (ty::FnDiverging, ty::FnDiverging) =>
                Ok(ty::FnDiverging),
            (a, b) =>
                Err(ty::terr_convergence_mismatch(
                    combine::expected_found(combiner, a != ty::FnDiverging, b != ty::FnDiverging))),
        });

        return Ok(ty::FnSig {inputs: inputs,
                             output: output,
                             variadic: a.variadic});


        fn argvecs<'tcx, C: Combine<'tcx>>(combiner: &C,
                                           a_args: &[Ty<'tcx>],
                                           b_args: &[Ty<'tcx>])
                                           -> cres<'tcx, Vec<Ty<'tcx>>>
        {
            if a_args.len() == b_args.len() {
                a_args.iter().zip(b_args.iter())
                    .map(|(a, b)| combiner.args(*a, *b)).collect()
            } else {
                Err(ty::terr_arg_count)
            }
        }
    }
}

impl<'tcx> HigherRankedCombineable<'tcx> for ty::TraitRef<'tcx> {
    fn super_combine<C:Combine<'tcx>>(combiner: &C,
                                      a: &ty::TraitRef<'tcx>,
                                      b: &ty::TraitRef<'tcx>)
                                      -> cres<'tcx, ty::TraitRef<'tcx>>
    {
        // Different traits cannot be related
        if a.def_id != b.def_id {
            Err(ty::terr_traits(
                combine::expected_found(combiner, a.def_id, b.def_id)))
        } else {
            let substs = try!(combiner.substs(a.def_id, &a.substs, &b.substs));
            Ok(ty::TraitRef { def_id: a.def_id,
                              substs: substs })
        }
    }
}

fn var_ids<'tcx, T: Combine<'tcx>>(combiner: &T,
                                   map: &FnvHashMap<ty::BoundRegion, ty::Region>)
                                   -> Vec<ty::RegionVid> {
    map.iter().map(|(_, r)| match *r {
            ty::ReInfer(ty::ReVar(r)) => { r }
            r => {
                combiner.infcx().tcx.sess.span_bug(
                    combiner.trace().origin.span(),
                    format!("found non-region-vid: {}", r).as_slice());
            }
        }).collect()
}

fn is_var_in_set(new_vars: &[ty::RegionVid], r: ty::Region) -> bool {
    match r {
        ty::ReInfer(ty::ReVar(ref v)) => new_vars.iter().any(|x| x == v),
        _ => false
    }
}

fn fold_regions_in<'tcx, T, F>(tcx: &ty::ctxt<'tcx>, value: &T, mut fldr: F) -> T where
    T: HigherRankedFoldable<'tcx>,
    F: FnMut(ty::Region, ty::DebruijnIndex) -> ty::Region,
{
    value.fold_contents(&mut ty_fold::RegionFolder::new(tcx, |region, current_depth| {
        // we should only be encountering "escaping" late-bound regions here,
        // because the ones at the current level should have been replaced
        // with fresh variables
        assert!(match region {
            ty::ReLateBound(..) => false,
            _ => true
        });

        fldr(region, ty::DebruijnIndex::new(current_depth))
    }))
}

impl<'a,'tcx> InferCtxtExt<'tcx> for InferCtxt<'a,'tcx> {
    fn tainted_regions(&self, snapshot: &CombinedSnapshot, r: ty::Region) -> Vec<ty::Region> {
        self.region_vars.tainted(&snapshot.region_vars_snapshot, r)
    }

    fn region_vars_confined_to_snapshot(&self,
                                        snapshot: &CombinedSnapshot)
                                        -> Vec<ty::RegionVid>
    {
        /*!
         * Returns the set of region variables that do not affect any
         * types/regions which existed before `snapshot` was
         * started. This is used in the sub/lub/glb computations. The
         * idea here is that when we are computing lub/glb of two
         * regions, we sometimes create intermediate region variables.
         * Those region variables may touch some of the skolemized or
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
         * algorith, we begin by replacing `'a` with a skolemized
         * variable `'0`. We then have `fn(_#0t) <: fn(&'0 int)`. This
         * can be made true by unifying `_#0t` with `&'0 int`. In the
         * process, we create a fresh variable for the skolemized
         * region, `'$0`, and hence we have that `_#0t == &'$0
         * int`. However, because `'$0` was created during the sub
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
            self.region_vars.vars_created_since_snapshot(&snapshot.region_vars_snapshot);

        let escaping_types =
            self.type_variables.borrow().types_escaping_snapshot(&snapshot.type_snapshot);

        let escaping_region_vars: FnvHashSet<_> =
            escaping_types
            .iter()
            .flat_map(|&t| ty_fold::collect_regions(self.tcx, &t).into_iter())
            .collect();

        region_vars.retain(|&region_vid| {
            let r = ty::ReInfer(ty::ReVar(region_vid));
            !escaping_region_vars.contains(&r)
        });

        debug!("region_vars_confined_to_snapshot: region_vars={} escaping_types={}",
               region_vars.repr(self.tcx),
               escaping_types.repr(self.tcx));

        region_vars
    }
}
