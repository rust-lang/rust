//! Helper routines for higher-ranked things. See the `doc` module at
//! the end of the file for details.

use super::combine::CombineFields;
use super::{HigherRankedType, InferCtxt, PlaceholderMap};

use crate::infer::CombinedSnapshot;
use crate::ty::relate::{Relate, RelateResult, TypeRelation};
use crate::ty::{self, Binder, TypeFoldable};
use crate::mir::interpret::ConstValue;

impl<'a, 'tcx> CombineFields<'a, 'tcx> {
    pub fn higher_ranked_sub<T>(
        &mut self,
        a: &Binder<T>,
        b: &Binder<T>,
        a_is_expected: bool,
    ) -> RelateResult<'tcx, Binder<T>>
    where
        T: Relate<'tcx>,
    {
        debug!("higher_ranked_sub(a={:?}, b={:?})", a, b);

        // Rather than checking the subtype relationship between `a` and `b`
        // as-is, we need to do some extra work here in order to make sure
        // that function subtyping works correctly with respect to regions
        //
        // Note: this is a subtle algorithm.  For a full explanation,
        // please see the large comment at the end of the file in the (inlined) module
        // `doc`.

        let span = self.trace.cause.span;

        return self.infcx.commit_if_ok(|snapshot| {
            // First, we instantiate each bound region in the supertype with a
            // fresh placeholder region.
            let (b_prime, placeholder_map) = self.infcx.replace_bound_vars_with_placeholders(b);

            // Next, we instantiate each bound region in the subtype
            // with a fresh region variable. These region variables --
            // but no other pre-existing region variables -- can name
            // the placeholders.
            let (a_prime, _) =
                self.infcx
                    .replace_bound_vars_with_fresh_vars(span, HigherRankedType, a);

            debug!("a_prime={:?}", a_prime);
            debug!("b_prime={:?}", b_prime);

            // Compare types now that bound regions have been replaced.
            let result = self.sub(a_is_expected).relate(&a_prime, &b_prime)?;

            self.infcx
                .leak_check(!a_is_expected, &placeholder_map, snapshot)?;

            debug!("higher_ranked_sub: OK result={:?}", result);

            Ok(ty::Binder::bind(result))
        });
    }
}

impl<'a, 'tcx> InferCtxt<'a, 'tcx> {
    /// Replaces all regions (resp. types) bound by `binder` with placeholder
    /// regions (resp. types) and return a map indicating which bound-region
    /// placeholder region. This is the first step of checking subtyping
    /// when higher-ranked things are involved.
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
        binder: &ty::Binder<T>,
    ) -> (T, PlaceholderMap<'tcx>)
    where
        T: TypeFoldable<'tcx>,
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

        let fld_c = |bound_var: ty::BoundVar, ty| {
            self.tcx.mk_const(
                ty::Const {
                    val: ConstValue::Placeholder(ty::PlaceholderConst {
                        universe: next_universe,
                        name: bound_var,
                    }),
                    ty,
                }
            )
        };

        let (result, map) = self.tcx.replace_bound_vars(binder, fld_r, fld_t, fld_c);

        debug!(
            "replace_bound_vars_with_placeholders(\
             next_universe={:?}, \
             binder={:?}, \
             result={:?}, \
             map={:?})",
            next_universe, binder, result, map,
        );

        (result, map)
    }

    /// See `infer::region_constraints::RegionConstraintCollector::leak_check`.
    pub fn leak_check(
        &self,
        overly_polymorphic: bool,
        placeholder_map: &PlaceholderMap<'tcx>,
        snapshot: &CombinedSnapshot<'_, 'tcx>,
    ) -> RelateResult<'tcx, ()> {
        self.borrow_region_constraints()
            .leak_check(self.tcx, overly_polymorphic, placeholder_map, snapshot)
    }
}
