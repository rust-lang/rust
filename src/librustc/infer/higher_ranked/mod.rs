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

use super::{InferCtxt,
            HigherRankedType};
use super::combine::CombineFields;

use ty::{self, Binder, TypeFoldable};
use ty::relate::{Relate, RelateResult, TypeRelation};

impl<'a, 'gcx, 'tcx> CombineFields<'a, 'gcx, 'tcx> {
    pub fn higher_ranked_sub<T>(&mut self,
                                param_env: ty::ParamEnv<'tcx>,
                                a: &Binder<T>,
                                b: &Binder<T>,
                                a_is_expected: bool)
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
        return self.infcx.commit_if_ok(|_snapshot| {
            let span = self.trace.cause.span;

            // First, we instantiate each bound region in the supertype with a
            // fresh concrete region.
            let b_prime =
                self.infcx.skolemize_late_bound_regions(b);

            // Second, we instantiate each bound region in the subtype with a fresh
            // region variable.
            let (a_prime, _) =
                self.infcx.replace_late_bound_regions_with_fresh_var(
                    span,
                    HigherRankedType,
                    a);

            debug!("a_prime={:?}", a_prime);
            debug!("b_prime={:?}", b_prime);

            // Compare types now that bound regions have been replaced.
            let result = self.sub(param_env, a_is_expected).relate(&a_prime, &b_prime)?;

            debug!("higher_ranked_sub: OK result={:?}", result);

            Ok(ty::Binder(result))
        });
    }
}

impl<'a, 'gcx, 'tcx> InferCtxt<'a, 'gcx, 'tcx> {
    /// Replace all regions bound by `binder` with skolemized regions and
    /// return a map indicating which bound-region was replaced with what
    /// skolemized region. This is the first step of checking subtyping
    /// when higher-ranked things are involved.
    ///
    /// See `README.md` for more details.
    pub fn skolemize_late_bound_regions<T>(&self,
                                           binder: &ty::Binder<T>)
                                           -> T
        where T : TypeFoldable<'tcx>
    {
        let (result, _) = self.tcx.replace_late_bound_regions(binder, |br| {
            self.universe.set(self.universe().subuniverse());
            self.tcx.mk_region(ty::ReSkolemized(self.universe(), br))
        });

        debug!("skolemize_bound_regions(binder={:?}, result={:?})",
               binder,
               result);

        result
    }
}
